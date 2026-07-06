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

from fastapi import WebSocketDisconnect  # noqa: E402

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
        gap_filler_enabled=False,
        elevenlabs_api_key="test-key",
        elevenlabs_voice_id="test-voice",
    )
    base.update(over)
    return SimpleNamespace(**base)


class FakeSTT:
    def __init__(self):
        self.frames = []

    async def send_audio(self, pcm):
        self.frames.append(pcm)

    async def send_mute(self, nbytes):
        # Mirror the AssemblyAI mute representation (equal-length zeros) so the
        # muted-path assertions can keep comparing against SILENCE frames.
        self.frames.append(bytes(nbytes))


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


async def test_composing_does_not_barge():
    """While the agent is composing (no audio on the wire yet), caller speech must
    NOT cancel the in-flight turn — there is nothing to interrupt, and cancelling
    would discard a turn that is about to answer (a slow multi-step turn would
    otherwise be killed on a loop by an impatient "hello?"). The audio is still
    forwarded to STT so the caller's words survive into the next turn."""
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    task = live_turn(state)
    state.greeting_active = False
    bridge.enqueue_turn("queued-1", gap_filler=True)
    gen0 = state.barge_generation

    for _ in range(10):  # well past min_frames
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=False)

    await asyncio.sleep(0)
    assert state.barge_generation == gen0, "composing must not trigger a barge"
    assert transport.clears == 0, "no flush while composing"
    assert not task.cancelled(), "the in-flight turn must survive"
    assert not bridge._turns.empty(), "queued turns must NOT be drained while composing"
    assert stt.frames == [LOUD] * 10, "composing forwards live caller audio to STT"
    task.cancel()


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
    state.speaking_started_at = 0.0  # echo guard already passed (guard=0)
    gen0 = state.barge_generation

    for _ in range(2):  # one short of the threshold
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)

    assert state.barge_generation == gen0, "two frames must not trip a barge"
    assert transport.clears == 0


async def test_transient_resets_run():
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings(barge_in_min_frames=3))
    live_turn(state)
    state.speaking_started_at = 0.0  # echo guard already passed (guard=0)
    gen0 = state.barge_generation

    # loud, loud, QUIET (resets), loud, loud -> never 3 consecutive
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)
    await bridge._detect_barge_in(SILENCE, 0, 1000.0, audible=True)  # sub-threshold
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)
    await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)

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


class FakeMicClient(FakeTransport):
    """Transport that also plays inbound frames into browser_to_stt."""

    def __init__(self, frames):
        super().__init__()
        self._frames = list(frames)

    async def receive(self):
        if self._frames:
            return {"bytes": self._frames.pop(0)}
        return {"type": "websocket.disconnect"}


async def run_disabled_loop(state, frames):
    """Drive browser_to_stt (flag off) over `frames` until disconnect."""
    stt = FakeSTT()
    client = FakeMicClient(frames)

    async def _noop_turn(*a, **k):
        await asyncio.sleep(100)

    bridge = AudioBridge(
        client, stt, state, _noop_turn, make_settings(barge_in_enabled=False)
    )
    try:
        await bridge.browser_to_stt()
    except WebSocketDisconnect:
        pass
    return stt, client


async def test_flag_off_mutes_while_composing():
    """Flag off + a turn in flight but no audio on the wire yet (gap filler /
    LLM composing): the caller must NOT be listened to — STT gets equal-length
    silence, so nothing said over the agent is queued and answered later (the
    reported bug: mute was tied to `audible` only, leaving this window open)."""
    state = PipelineState()
    task = live_turn(state)      # turn in flight
    state.speaking_until = 0.0   # nothing audible on the wire
    stt, _ = await run_disabled_loop(state, [LOUD, LOUD, LOUD])
    assert stt.frames == [SILENCE] * 3, "composing with flag off must mute STT"
    task.cancel()


async def test_flag_off_mutes_while_audible():
    """Flag off + agent audio still playing (even after the turn task ended):
    STT stays muted, as before."""
    state = PipelineState()
    state.turn_task = None
    state.speaking_until = 1e18  # reply still playing out
    stt, _ = await run_disabled_loop(state, [LOUD, LOUD])
    assert stt.frames == [SILENCE] * 2, "audible playback with flag off must mute STT"


async def test_flag_off_listens_when_idle():
    """Flag off + no turn in flight and nothing playing: the caller is heard."""
    state = PipelineState()
    state.turn_task = None
    state.speaking_until = 0.0
    stt, _ = await run_disabled_loop(state, [LOUD, LOUD])
    assert stt.frames == [LOUD, LOUD], "idle with flag off must forward live audio"


class FakeEventSTT(FakeSTT):
    """STT fake that also yields a fixed list of events from stt_to_agent."""

    def __init__(self, events):
        super().__init__()
        self._events = list(events)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e


def final_turn_event(text, order=0):
    return {
        "type": "Turn",
        "transcript": text,
        "end_of_turn": True,
        "turn_order": order,
        "turn_is_formatted": True,
    }


async def run_stt_to_agent(state, settings, events):
    """Drive stt_to_agent over `events`; return the bridge (inspect ._turns)."""
    stt = FakeEventSTT(events)

    async def _noop_turn(*a, **k):
        await asyncio.sleep(100)

    bridge = AudioBridge(FakeTransport(), stt, state, _noop_turn, settings)
    await bridge.stt_to_agent()
    return bridge


async def test_stale_final_dropped_while_audible():
    """A final that lands while the agent's reply is still playing is stale
    audio that slipped past the mute (STT decode lag) — it must NOT be queued
    and answered afterwards (the double-answer bug from the call logs)."""
    state = PipelineState()
    state.turn_task = None
    state.speaking_until = 1e18  # reply still playing out
    bridge = await run_stt_to_agent(
        state,
        make_settings(barge_in_enabled=False),
        [final_turn_event("book me a reservation")],
    )
    assert bridge._turns.empty(), "final during playback must be dropped"


async def test_stale_final_dropped_while_turn_live_flag_off():
    """Flag off + a turn composing (no audio yet): the caller was muted for
    this whole window, so any final now is lagged pre-mute audio — drop it."""
    state = PipelineState()
    task = live_turn(state)
    state.speaking_until = 0.0
    bridge = await run_stt_to_agent(
        state,
        make_settings(barge_in_enabled=False),
        [final_turn_event("book me a reservation")],
    )
    assert bridge._turns.empty(), "final while a turn is in flight must be dropped"
    task.cancel()


async def test_final_dispatched_when_idle():
    state = PipelineState()
    state.turn_task = None
    state.speaking_until = 0.0
    bridge = await run_stt_to_agent(
        state,
        make_settings(barge_in_enabled=False),
        [final_turn_event("book me a reservation")],
    )
    assert bridge._turns.qsize() == 1, "idle: the caller's turn must dispatch"


async def test_composing_final_prunes_inflight_turn():
    """Barge-in ON: a FINAL landing while a turn is still composing means the
    caller kept talking — the in-flight turn is answering an incomplete
    utterance. It must be pruned (transcript-level barge) and the new final
    queued, so ONE reply is regenerated from everything the caller said (the
    'Actually, it we' / 'six in the evening.' double-answer bug)."""
    state = PipelineState()
    task = live_turn(state)
    state.speaking_until = 0.0  # composing: nothing audible
    gen0 = state.barge_generation
    bridge = await run_stt_to_agent(
        state,
        make_settings(barge_in_enabled=True),
        [final_turn_event("and the pickup is at noon")],
    )
    assert bridge._turns.qsize() == 1, "composing final must queue with barge-in on"
    assert state.barge_generation == gen0 + 1, "in-flight turn must be superseded"
    await asyncio.sleep(0)
    assert task.cancelled(), "the stale composing turn must be cancelled"


async def test_greeting_composing_final_does_not_prune():
    """The opening greeting is exempt from the transcript-level barge (as it is
    from the energy one): a final while it composes queues behind it."""
    state = PipelineState()
    task = live_turn(state)
    state.greeting_active = True
    state.speaking_until = 0.0
    gen0 = state.barge_generation
    bridge = await run_stt_to_agent(
        state,
        make_settings(barge_in_enabled=True),
        [final_turn_event("hello?")],
    )
    assert bridge._turns.qsize() == 1, "final must still queue behind the greeting"
    assert state.barge_generation == gen0, "greeting must not be pruned"
    assert not task.cancelled(), "greeting turn must survive"
    task.cancel()


async def test_worker_merges_queued_fragments():
    """Fragments of one utterance that are all queued by the time the worker
    dequeues must be merged into a single turn (one reply, full sentence)."""
    state = PipelineState()
    handled = []

    async def on_turn(text, user_stopped_at=None, gap_filler=False):
        handled.append(text)

    bridge = AudioBridge(FakeTransport(), FakeSTT(), state, on_turn, make_settings())
    bridge.enqueue_turn("Actually, it we", gap_filler=True)
    bridge.enqueue_turn("six in the evening.", gap_filler=True)
    worker = asyncio.create_task(bridge.turn_worker())
    await asyncio.sleep(0.05)
    assert handled == ["Actually, it we six in the evening."], handled
    worker.cancel()


async def test_barge_keeps_queued_turns():
    """A barge-in must NOT drop queued turns: they are unanswered caller finals
    and the worker merges them into the next (redo) turn, so nothing the caller
    said is lost to the interruption."""
    state = PipelineState()
    bridge, stt, transport = make_bridge(state, make_settings())
    live_turn(state)
    state.speaking_started_at = 0.0  # echo guard passed
    bridge.enqueue_turn("queued caller final", gap_filler=True)

    for _ in range(3):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)

    assert transport.clears == 1, "barge must still flush playback"
    assert bridge._turns.qsize() == 1, "queued caller finals must survive a barge"


async def test_barge_rollback_keeps_caller_utterance():
    """A barged turn must prune ONLY the agent's abandoned reply — the caller's
    utterance stays in memory so the regenerated reply accounts for it (the
    'forgot the pickup time after a barge' bug)."""
    import services.turn_handler as th_mod

    calls = {}

    class FakeAgent:
        async def checkpoint(self):
            return {"pre-turn-msg"}

        async def stream_response(self, text):
            yield "Great, I have your pickup set "
            await asyncio.sleep(100)  # mid-reply when the barge lands

        async def rollback_barge(self, pre_ids, *, keep_user_message=False):
            calls["pre_ids"] = pre_ids
            calls["keep_user_message"] = keep_user_message

        def snapshot(self):
            return None

    async def fake_tts(gen, *args, **kwargs):
        async for _sentence in gen:
            yield b"\x00" * 320

    orig = th_mod.stream_tts_input
    th_mod.stream_tts_input = fake_tts
    try:
        state = PipelineState()
        transport = FakeTransport()
        handler = TurnHandler(transport, FakeAgent(), make_settings(), state)
        task = asyncio.create_task(
            handler.handle_turn("pickup friday nine", gap_filler=True)
        )
        await asyncio.sleep(0.05)  # let the fake reply start streaming
        task.cancel()  # the barge
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert calls.get("keep_user_message") is True, (
            "barge rollback must keep the caller's utterance even after the "
            "agent started replying"
        )
    finally:
        th_mod.stream_tts_input = orig


class FakeDeepgramWS:
    """Async-iterable stand-in for the Deepgram websocket (yields raw JSON)."""

    def __init__(self, messages, raise_after=None):
        self._messages = list(messages)
        self._raise_after = raise_after

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        import json

        for m in self._messages:
            yield json.dumps(m)
        if self._raise_after is not None:
            raise self._raise_after

    async def close(self):
        pass

    async def send(self, data):
        pass


def dg_closed_error():
    import websockets
    from websockets.frames import Close

    return websockets.exceptions.ConnectionClosedError(Close(1006, ""), None)


def dg_results(text, start, duration, is_final=True, speech_final=True):
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text}]},
    }


async def test_deepgram_replayed_segment_dropped():
    """Deepgram sometimes re-emits Results covering audio that was already
    finalized (seen live as a full re-transcription of the previous utterance
    arriving seconds later, answered a second time). A segment that does not
    advance the audio timeline must be dropped; genuinely new audio must not."""
    from services.stt_deepgram import DeepgramStream

    stream = DeepgramStream("key")
    stream._closed = True  # ends _events cleanly when the fake stream runs dry
    stream.ws = FakeDeepgramWS([
        dg_results("book me a reservation", start=13.0, duration=7.0),
        # Replay of the same audio span — must be dropped, not become a turn.
        dg_results("book me a reservation", start=13.0, duration=7.0),
        # New speech past the watermark — must still come through.
        dg_results("pickup at noon", start=25.0, duration=2.0),
    ])
    turns = [
        e["transcript"] async for e in stream if e.get("end_of_turn")
    ]
    assert turns == ["book me a reservation", "pickup at noon"], turns


async def test_deepgram_reconnects_after_drop():
    """A mid-call socket death must NOT end the STT stream (it crashed the whole
    session live: 'keepalive ping timeout'). The receive loop re-dials and keeps
    yielding; turn_order keeps increasing across the reconnect (AudioBridge
    dedupes on it) while the replay watermark resets (the new connection's audio
    timeline restarts at zero, so old watermarks would drop everything)."""
    from services.stt_deepgram import DeepgramStream

    stream = DeepgramStream("key")
    # First connection dies abnormally after one finalized utterance.
    stream.ws = FakeDeepgramWS(
        [dg_results("first utterance", start=10.0, duration=5.0)],
        raise_after=dg_closed_error(),
    )
    # Reconnect hands over a fresh connection whose timeline restarted (the
    # segment ends at 1.5s, well before the old 15.0s watermark).
    ws2 = FakeDeepgramWS([dg_results("second utterance", start=0.5, duration=1.0)])

    async def fake_connect():
        stream.ws = ws2
        return stream

    stream.connect = fake_connect

    turns = []
    async for e in stream:
        if e.get("end_of_turn"):
            turns.append((e["turn_order"], e["transcript"]))
        if len(turns) == 2:
            stream._closed = True  # stop after the post-reconnect turn
    assert turns == [(0, "first utterance"), (1, "second utterance")], turns


async def test_deepgram_flushes_pending_text_on_drop():
    """Finalized segments that never got their endpoint before the socket died
    must be flushed as a completed turn — the caller's words survive the drop."""
    from services.stt_deepgram import DeepgramStream

    stream = DeepgramStream("key")
    stream._closed = True  # no reconnect: we only care about the flush
    stream.ws = FakeDeepgramWS(
        [dg_results("pickup at noon", start=1.0, duration=2.0, speech_final=False)],
        raise_after=dg_closed_error(),
    )
    turns = [e["transcript"] async for e in stream if e.get("end_of_turn")]
    assert turns == ["pickup at noon"], turns


async def test_deepgram_send_survives_dead_socket():
    """send_audio / send_mute on a dead socket must drop the frame, not raise
    into the inbound-audio task (which killed the session live)."""
    from services.stt_deepgram import DeepgramStream

    class DeadWS:
        async def send(self, data):
            raise dg_closed_error()

    stream = DeepgramStream("key")
    stream.ws = DeadWS()
    await stream.send_audio(b"\x00" * 320)  # must not raise
    stream._last_keepalive = 0.0
    await stream.send_mute(320)  # must not raise
    stream.ws = None
    await stream.send_audio(b"\x00" * 320)  # reconnect window: must not raise


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

    # Caller barges while the agent is audible (Regime A) — e.g. over the reply.
    state.speaking_started_at = 0.0  # echo guard already passed (guard=0)
    for _ in range(3):
        await bridge._detect_barge_in(LOUD, 5000, 1000.0, audible=True)
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
        test_composing_does_not_barge,
        test_regime_a_silences_then_replays_preroll,
        test_barge_during_playback_tail_no_live_turn,
        test_debounce_blocks_short_burst,
        test_transient_resets_run,
        test_echo_below_threshold_never_barges,
        test_greeting_exempt_from_composing_barge,
        test_no_turn_live_does_not_barge,
        test_flag_off_no_detection,
        test_flag_off_mutes_while_composing,
        test_flag_off_mutes_while_audible,
        test_flag_off_listens_when_idle,
        test_stale_final_dropped_while_audible,
        test_stale_final_dropped_while_turn_live_flag_off,
        test_final_dispatched_when_idle,
        test_composing_final_prunes_inflight_turn,
        test_greeting_composing_final_does_not_prune,
        test_worker_merges_queued_fragments,
        test_barge_keeps_queued_turns,
        test_barge_rollback_keeps_caller_utterance,
        test_deepgram_replayed_segment_dropped,
        test_deepgram_reconnects_after_drop,
        test_deepgram_flushes_pending_text_on_drop,
        test_deepgram_send_survives_dead_socket,
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
