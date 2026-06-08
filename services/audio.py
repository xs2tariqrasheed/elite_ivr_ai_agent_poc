"""Audio transcoding helpers.

Twilio Media Streams carry G.711 μ-law at 8 kHz, but AssemblyAI's
universal-streaming model is built around 16 kHz linear PCM (feeding it raw
8 kHz μ-law yields no transcription). We decode μ-law to PCM16 and upsample to
16 kHz so the phone path uses the same STT format as the browser path.

`audioop` was removed in Python 3.13, so the G.711 decode is done with numpy.
"""
import numpy as np


def _build_mulaw_table() -> np.ndarray:
    """256-entry lookup from a μ-law byte to its signed 16-bit PCM sample."""
    table = np.empty(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        table[i] = -sample if sign else sample
    return table


_MULAW_TO_PCM16 = _build_mulaw_table()


def mulaw8k_to_pcm16_16k(mulaw: bytes) -> bytes:
    """Decode μ-law@8 kHz to little-endian PCM16 and linearly upsample to 16 kHz."""
    if not mulaw:
        return b""
    samples = _MULAW_TO_PCM16[np.frombuffer(mulaw, dtype=np.uint8)].astype(np.float32)
    midpoints = np.empty_like(samples)
    midpoints[:-1] = (samples[:-1] + samples[1:]) * 0.5
    midpoints[-1] = samples[-1]
    upsampled = np.empty(samples.size * 2, dtype=np.float32)
    upsampled[0::2] = samples
    upsampled[1::2] = midpoints
    return upsampled.astype(np.int16).tobytes()
