"""In-memory gap-filler audio played while the agent processes a caller's turn.

The agent's reply has a few seconds of latency (STT finalize -> LLM -> TTS). To
avoid dead air, we play a short pre-recorded acknowledgement ("one moment...")
the instant the caller stops speaking, while the LLM works in the background.

The clips ship as MP3 (services/gap_fillers/*.mp3) but the pipeline sends raw
audio on the wire — PCM16@16k to the browser, μ-law@8k to Twilio (see
configs.AudioFormat). MP3 can't be sent as-is, so each clip is decoded once at
startup (via ffmpeg) into both wire formats and cached in memory, ready to send
with zero per-call work.
"""
import logging
import random
import subprocess
from pathlib import Path

log = logging.getLogger("voice")

_DIR = Path(__file__).parent / "gap_fillers"

# ffmpeg output args per pipeline wire format (see configs.AudioFormat). Mono,
# matching sample rate, headerless raw frames — identical to what TTS emits.
_FFMPEG_ARGS: dict[str, list[str]] = {
    "pcm_16000": ["-ar", "16000", "-ac", "1", "-f", "s16le"],
    "ulaw_8000": ["-ar", "8000", "-ac", "1", "-f", "mulaw"],
}

# {output_format: [decoded raw audio bytes per clip]}. Populated by load().
_CLIPS: dict[str, list[bytes]] = {fmt: [] for fmt in _FFMPEG_ARGS}


def _decode(path: Path, output_format: str) -> bytes:
    """Decode one MP3 to raw `output_format` bytes via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *_FFMPEG_ARGS[output_format], "-"],
        capture_output=True,
        check=True,
    )
    return proc.stdout


def load() -> None:
    """Decode every gap-filler MP3 into all wire formats and cache in memory.

    Best-effort: a clip that fails to decode (e.g. ffmpeg missing) is skipped and
    logged rather than crashing startup — the pipeline simply plays no gap filler.
    """
    for fmt in _CLIPS:
        _CLIPS[fmt].clear()
    files = sorted(_DIR.glob("*.mp3"))
    for path in files:
        for fmt in _FFMPEG_ARGS:
            try:
                audio = _decode(path, fmt)
            except (OSError, subprocess.CalledProcessError) as exc:
                log.warning("Gap filler decode failed (%s, %s): %s", path.name, fmt, exc)
                continue
            if audio:
                _CLIPS[fmt].append(audio)
    log.info(
        "Loaded %d gap fillers (formats: %s)",
        len(files),
        ", ".join(f"{fmt}={len(clips)}" for fmt, clips in _CLIPS.items()),
    )


def random_gap_filler(output_format: str) -> bytes | None:
    """Return a random pre-decoded clip for `output_format`, or None if none cached."""
    clips = _CLIPS.get(output_format)
    if not clips:
        return None
    return random.choice(clips)
