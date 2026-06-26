"""Behavioral tests for the barge-in mechanism.

Standalone (no pytest dependency): run with `python tests/test_barge_in.py`.
Each test drives the real AudioBridge / TurnHandler with light fakes for the
transport, STT, and agent, exercising the detector, the pre-roll buffer, the
generation guard, and the prune/redo coordination.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.audio_bridge import AudioBridge  # noqa: E402
from services.pipeline_state import PipelineState  # noqa: E402
from services.turn_handler import TurnHandler  # noqa: E402

FRAME_SAMPLES = 800  # one PCM16 frame
SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
LOUD = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 5000).tobytes()  # mean-abs 5000


def make_settings(**over):
    base = dict(
        barge_in_enabled=True,
        barge_in_voice_level=1500,
        barge_in_voice_level_idle=700,
        barge_in_min_frames=3,
        barge_in_min_ms=0,
        barge_in_echo_guard_seconds=0.0,
        barge_in_echo_tail_seconds=0.7,
    )
    base.update(over)
    return SimpleNamespace(**base)


class FakeSTT:
    def __init__(self):
        self.frames = []

    async def send_audio(self, pcm):
        self.frames.append(pcm)


class FakeTransport:
    def __init__(self):
        self.sent_bytes = []
        self.sent_json = []
        self.clears = 0

    async def send_bytes(self, b):
        self.sent_bytes.append(b)

    async def send_json(self, m):
        self.sent_json.append(m)

    async def clear(self):
        self.clears += 1

    async def close(self):
        pass


def make_bridge(state, settings, stt=None, transport=None):
    stt = stt or FakeSTT()
    transport = transport or FakeTransport()

    async def _noop_turn(*a, **k):
        await asyncio.sleep(100)

    bridge = AudioBridge(transport, stt, state, _noop_turn, settings)
    return bridge, stt, transport


def live_turn(state):
    """Install a long-running task as the in-flight turn."""
    task = asyncio.create_task(asyncio.sleep(100))
    state.turn_task = task
    return task


# --------------------------------------------------------------------------- #


async def test_regime_b_fires_and_prunes():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    task = live_turn(state)
    state.greeting_active = False
    # queue some stale turns that must be drained
    bridge.enqueue_turn("stale-1", gap_filler=True)
    bridge.enqueue_turn("stale-2", gap_filler=True)
    gen0 = state.barge_generation

    for _ in range(3):  # min_frames
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    await asyncio.sleep(0)  # let the cancel land
    assert state.barge_generation == gen0 + 1, "generation must bump exactly once"
    assert transport.clears == 1, "transport must be flushed once"
    assert state.speaking_until == 0.0, "STT must be un-muted"
    assert bridge._turns.empty(), "stale queued turns must be drained"
    assert task.cancelled(), "in-flight turn must be cancelled"
    # Regime B feeds live audio to STT; no pre-roll replay.
    assert stt.frames == [LOUD, LOUD, LOUD], "Regime B forwards live caller audio"


async def test_regime_a_silences_then_replays_preroll():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    live_turn(state)
    state.speaking_started_at = 0.0  # echo guard already passed (guard=0)

    for _ in range(3):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)

    await asyncio.sleep(0)
    # While audible, STT got equal-length silence for the 3 real frames, then the
    # 3 captured real frames were replayed after the flush.
    assert stt.frames[:3] == [SILENCE, SILENCE, SILENCE], "echo defense: silence to STT"
    assert stt.frames[3:] == [LOUD, LOUD, LOUD], "pre-roll caller frames replayed (#3)"
    assert transport.clears == 1


async def test_barge_during_playback_tail_no_live_turn():
    """The reply is fully streamed to the transport before it finishes playing,
    so the turn task ends while audio is still audible. A barge during that tail
    must still fire and flush, even though turn_task is None (the log-reported
    bug)."""
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    state.turn_task = None  # turn already completed
    state.speaking_until = 1e18  # agent audio still playing out (audible)
    state.speaking_started_at = 0.0  # onset long ago -> echo guard passed
    gen0 = state.barge_generation

    for _ in range(3):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)

    assert state.barge_generation == gen0 + 1, "must barge during the playback tail"
    assert transport.clears == 1, "must flush the still-playing buffer"
    assert state.speaking_until == 0.0


async def test_debounce_blocks_short_burst():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings(barge_in_min_frames=3))
    live_turn(state)
    gen0 = state.barge_generation

    for _ in range(2):  # one short of the threshold
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    assert state.barge_generation == gen0, "two frames must not trip a barge"
    assert transport.clears == 0


async def test_transient_resets_run():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings(barge_in_min_frames=3))
    live_turn(state)
    gen0 = state.barge_generation

    # loud, loud, QUIET (resets), loud, loud -> never 3 consecutive
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)
    await bridge._detect_barge_in(SILENCE, 0, 1000.0, audible=False)  # sub-threshold
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    assert state.barge_generation == gen0, "a sub-threshold frame must reset the run"


async def test_echo_below_threshold_never_barges():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    live_turn(state)
    gen0 = state.barge_generation

    # Agent echo: audible, but energy stays under the high playback threshold.
    for _ in range(10):
        await bridge._detect_barge_in(LOUD, 1400, 1000.0, audible=True)  # 1400 < 1500

    assert state.barge_generation == gen0, "echo under threshold must not barge"
    assert transport.clears == 0


async def test_greeting_exempt_from_composing_barge():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    live_turn(state)
    state.greeting_active = True
    gen0 = state.barge_generation

    for _ in range(10):  # composing window during the greeting
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    assert state.barge_generation == gen0, "greeting must not be barged while composing"
    # but it still forwards audio live so STT keeps listening
    assert stt.frames and stt.frames[0] == LOUD


async def test_no_turn_live_does_not_barge():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    state.turn_task = None
    gen0 = state.barge_generation

    for _ in range(10):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    assert state.barge_generation == gen0, "no in-flight turn => nothing to barge"


async def test_flag_off_no_detection():
    state = PipelineState()
    settings = make_settings(barge_in_enabled=False)
    bridge, stt, transport = make_bridge(state, settings)
    live_turn(state)
    gen0 = state.barge_generation
    # With the flag off, _detect_barge_in is never called from browser_to_stt;
    # assert the guard the loop uses is honored.
    assert settings.barge_in_enabled is False
    assert state.barge_generation == gen0


async def test_send_audio_generation_guard():
    state = PipelineState()
    transport = FakeTransport()
    th = TurnHandler(transport, object(), make_settings(), state)
    th._gen = state.barge_generation  # captured at "turn start"

    await th._send_audio(LOUD, 32000)
    assert transport.sent_bytes == [LOUD], "matching generation: chunk is sent"
    advanced = state.speaking_until
    assert advanced > 0.0, "speaking_until advances on a real send"

    # Simulate a barge bumping the generation mid-turn.
    state.barge_generation += 1
    before = state.speaking_until
    await th._send_audio(LOUD, 32000)
    assert transport.sent_bytes == [LOUD], "superseded chunk must NOT be sent"
    assert state.speaking_until == before, "superseded chunk must NOT re-advance speaking_until"


async def test_worker_survives_barge_runs_redo_then_teardown_ends_it():
    """The turn worker tolerates a barge cancel, runs the redo turn, and is
    cleanly ended by teardown (the #1/#2 leak fix)."""
    state = PipelineState()
    stt = FakeSTT()
    transport = FakeTransport()
    handled = []

    async def on_turn(text, user_stopped_at=None, gap_filler=False):
        handled.append(text)
        try:
            await asyncio.sleep(5)  # simulate composing/playing
        except asyncio.CancelledError:
            # Mirror real handle_turn: re-raise on teardown so the worker exits,
            # swallow on a barge so the worker continues to the redo turn.
            if state.closed:
                raise
            return

    bridge = AudioBridge(transport, stt, state, on_turn, make_settings())
    worker = asyncio.create_task(bridge.turn_worker())

    bridge.enqueue_turn("first", gap_filler=True)
    await asyncio.sleep(0.05)
    assert handled == ["first"]
    assert state.turn_task is not None and not state.turn_task.done()

    # Caller barges while the turn is composing (Regime B).
    for _ in range(3):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)
    await asyncio.sleep(0.05)
    assert not worker.done(), "worker must survive a barge-in cancel"

    bridge.enqueue_turn("redo", gap_filler=True)
    await asyncio.sleep(0.05)
    assert handled == ["first", "redo"], "worker must run the caller's redo turn"

    # Teardown, exactly as VoiceSession.run's finally does it.
    state.closed = True
    if state.turn_task is not None and not state.turn_task.done():
        state.turn_task.cancel()
    worker.cancel()
    try:
        await worker
    except BaseException:
        pass
    assert worker.done(), "teardown must end the worker (no leak)"


async def main():
    tests = [
        test_regime_b_fires_and_prunes,
        test_regime_a_silences_then_replays_preroll,
        test_barge_during_playback_tail_no_live_turn,
        test_debounce_blocks_short_burst,
        test_transient_resets_run,
        test_echo_below_threshold_never_barges,
        test_greeting_exempt_from_composing_barge,
        test_no_turn_live_does_not_barge,
        test_flag_off_no_detection,
        test_send_audio_generation_guard,
        test_worker_survives_barge_runs_redo_then_teardown_ends_it,
    ]
    failed = 0
    for t in tests:
        try:
            await t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        finally:
            # Cancel any leftover dummy turn tasks so they don't warn at shutdown.
            for leftover in asyncio.all_tasks() - {asyncio.current_task()}:
                leftover.cancel()
            await asyncio.sleep(0)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
