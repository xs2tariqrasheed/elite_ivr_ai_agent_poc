"""Audio conversion helpers + simple voice-activity detection.

Twilio Media Streams send caller audio as base64-encoded G.711 mu-law
at 8 kHz, mono.  faster-whisper expects mono float32 PCM at 16 kHz.
The helpers here translate between those two worlds and detect when
the caller has stopped speaking.
"""

import audioop
import logging
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def mulaw_to_pcm16_8k(mulaw_bytes: bytes) -> bytes:
    """Convert mu-law 8 kHz audio to signed 16-bit PCM 8 kHz."""
    return audioop.ulaw2lin(mulaw_bytes, 2)


def pcm16_8k_to_pcm16_16k(pcm16_8k: bytes) -> bytes:
    """Up-sample 16-bit PCM 8 kHz mono → 16 kHz mono."""
    converted, _state = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)
    return converted


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """Convert signed 16-bit PCM to float32 in [-1.0, 1.0]."""
    if not pcm16:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def mulaw_to_float32_16k(mulaw_bytes: bytes) -> np.ndarray:
    """One-shot: Twilio mu-law @ 8 kHz  →  float32 PCM @ 16 kHz mono."""
    pcm16_8k = mulaw_to_pcm16_8k(mulaw_bytes)
    pcm16_16k = pcm16_8k_to_pcm16_16k(pcm16_8k)
    return pcm16_to_float32(pcm16_16k)


# ---------------------------------------------------------------------------
# Voice activity detection (used to know when the caller stopped talking)
#
# We previously used webrtcvad mode 3 (strict). In practice, on real Twilio
# G.711 mu-law streams it flagged ~95% of frames as "voiced" even during
# absolute silence, which pinned the tail-silence check at ~900/900 ms and
# made listening never end. Energy (RMS) on the decoded PCM is far more
# predictable on phone audio: Twilio's comfort noise sits well below the
# threshold while real speech sits well above it.
# ---------------------------------------------------------------------------

_VAD_SAMPLE_RATE = 8000

# SENSITIVE CONFIG original was 20
_VAD_FRAME_MS = 40
_VAD_FRAME_BYTES = int(_VAD_SAMPLE_RATE * (_VAD_FRAME_MS / 1000.0)) * 2  # 320

# Two thresholds on the int16 RMS scale (0..32767):
#
#   _VOICED_RMS       ≈ -36 dBFS. Floor for "this frame contains SOMETHING";
#                     used to decide ``speech_started``. Comfortably above
#                     Twilio's comfort-noise floor (<150 on a clean line) but
#                     low enough to catch even quiet callers.
#
#   _SPEECH_TAIL_RMS  ≈ -29 dBFS. "Clearly speech-level energy". Used to
#                     decide ``speech_ended``: if the LOUDEST frame in the
#                     silence window is below this, there is no speech in
#                     that window.
#
# The tail check uses the PEAK (max) tail RMS rather than a count of
# above-threshold frames. This makes the silence decision robust to calls
# with a line-noise floor that sits just above _VOICED_RMS — such noise
# never peaks anywhere near speech-level energy, so the tail is correctly
# classified as silent and listening ends. The previous count-based check
# failed exactly in that case: every frame counted as "voiced", the tail
# stayed pinned at 900/900 ms, and _listen looped forever.
_VOICED_RMS = 500.0
_SPEECH_TAIL_RMS = 1200.0

# Last three ``tail_rms[-5:]`` snapshots from consecutive
# :func:`tail_last_five_stable_last_three_updates` calls (rolling window).
_TAIL_LAST_FIVE_HISTORY: Deque[Tuple[float, ...]] = deque(maxlen=3)


def reset_tail_last_five_stability_history() -> None:
    """Clear history used by :func:`tail_last_five_stable_last_three_updates`."""
    _TAIL_LAST_FIVE_HISTORY.clear()


def tail_last_five_stable_last_three_updates(tail_rms: List[float]) -> bool:
    """True if the last three received tail windows share the same final five RMS samples.

    Each call records ``tail_rms[-5:]`` (the same slice shape used when deriving
    ``tail_peak_rms = max(tail_rms)`` from the silence window). Returns ``True``
    only once three values have been recorded and all three five-sample tails are
    equal element-wise; otherwise ``False``.
    """
    if not tail_rms:
        return False
    key = tuple(tail_rms[-5:])
    _TAIL_LAST_FIVE_HISTORY.append(key)
    if len(_TAIL_LAST_FIVE_HISTORY) < 5:
        return False
    first = _TAIL_LAST_FIVE_HISTORY[0]
    return all(row == first for row in _TAIL_LAST_FIVE_HISTORY)


def _frame_rms(frame: bytes) -> float:
    """RMS amplitude of one 20 ms PCM16 frame on the int16 scale [0, 32767]."""
    if not frame:
        return 0.0
    arr = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _rms_per_frame(pcm16: bytes) -> list:
    """Per-frame RMS values for 20 ms frames of ``pcm16``."""
    values = []
    for off in range(0, len(pcm16) - _VAD_FRAME_BYTES + 1, _VAD_FRAME_BYTES):
        frame = pcm16[off : off + _VAD_FRAME_BYTES]
        values.append(_frame_rms(frame))
    return values


def _voiced_frames(pcm16: bytes) -> list:
    """Classify each 20 ms frame of ``pcm16`` as voiced (True) or silent (False).

    Uses RMS energy rather than webrtcvad because the latter is unreliable
    on Twilio phone audio (reports nearly all frames as voiced, including
    genuine silence).
    """
    return [r > _VOICED_RMS for r in _rms_per_frame(pcm16)]


def _has_voiced_run(flags, min_run_ms: int) -> bool:
    """True if ``flags`` contains at least one consecutive voiced run >= min_run_ms."""
    needed = max(1, (min_run_ms // _VAD_FRAME_MS))
    run = 0
    for v in flags:
        run = run + 1 if v else 0
        if run >= needed:
            return True
    return False


def detect_speech_boundaries(
    mulaw_bytes: bytes,
    silence_after_speech_ms: int = 1200,
    min_consecutive_speech_ms: int = 200,
) -> Tuple[bool, bool]:
    """Decide whether the caller has started and finished speaking.

    Returns ``(speech_started, speech_ended)``.

    ``speech_started`` becomes True once any consecutive run of
    above-``_VOICED_RMS`` frames reaches ``min_consecutive_speech_ms`` —
    a loose threshold that catches even quiet speech.

    ``speech_ended`` becomes True once speech has started AND the LOUDEST
    frame in the most recent ``silence_after_speech_ms`` of audio is below
    ``_SPEECH_TAIL_RMS`` (the "clearly speech-level energy" threshold).
    Using the tail peak rather than counting voiced frames keeps silence
    detection working on lines whose noise floor sits above ``_VOICED_RMS``:
    such noise never peaks near speech energy, so the tail is correctly
    classified as silent.
    """
    if not mulaw_bytes:
        return False, False

    pcm16 = mulaw_to_pcm16_8k(mulaw_bytes)
    rms_values = _rms_per_frame(pcm16)
    if not rms_values:
        return False, False

    flags = [r > _VOICED_RMS for r in rms_values]
    speech_started = _has_voiced_run(flags, min_consecutive_speech_ms)
    if not speech_started:
        return False, False

    window_frames = max(1, silence_after_speech_ms // _VAD_FRAME_MS)

    tail_rms = rms_values[-window_frames:]
    tail_peak_rms = max(tail_rms)
    speech_ended = tail_peak_rms < _SPEECH_TAIL_RMS

    # glitch in twilio, sending same audio chunk multiple times.
    if (
        tail_last_five_stable_last_three_updates(tail_rms)
        and speech_started
        and not speech_ended
    ):
        speech_ended = True

    if len(rms_values) < window_frames and not speech_ended:
        return True, False

    # tail_rms = rms_values[-window_frames:]
    # tail_peak_rms = max(tail_rms)
    # speech_ended = tail_peak_rms < _SPEECH_TAIL_RMS
    logger.debug(
        "vad: tail_peak_rms=%.0f speech_floor=%.0f",
        tail_peak_rms,
        _SPEECH_TAIL_RMS,
    )
    return True, speech_ended
