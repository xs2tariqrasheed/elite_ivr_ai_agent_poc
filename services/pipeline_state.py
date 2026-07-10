"""Mutable state shared across the pipeline components for one connection."""
import asyncio
from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Mutable state shared across all pipeline components for one connection."""

    turn_task: asyncio.Task | None = None
    closed: bool = False
    last_partial_at: float | None = None
    # Highest AssemblyAI turn_order already dispatched, so the punctuated
    # follow-up event for the same turn isn't dispatched a second time.
    last_dispatched_turn: int = -1
    # monotonic() time at which the agent's audio will finish playing. Inbound
    # caller audio is muted to STT until then (+ an echo tail) so the agent's
    # own voice can't bleed back in and corrupt / stall turn detection. Reset to
    # 0.0 on barge-in so STT un-mutes immediately for the caller's new utterance.
    speaking_until: float = 0.0

    # monotonic() time inbound caller energy last exceeded the idle voice
    # threshold while the agent was NOT audible (composing / between turns).
    # Feeds the pre-speak quiet gate (TurnHandler._hold_for_quiet_line): a
    # reply's first audio frame is held while this is fresh, so the agent never
    # starts talking over a caller who is mid-sentence. Regime-A (audible)
    # frames never update it — the agent's own echo would keep it permanently
    # fresh and hold every reply hostage.
    last_voice_at: float = 0.0

    # ----- Barge-in coordination ---------------------------------------------
    # monotonic() time the current contiguous playback began, for the echo-onset
    # guard (set by TurnHandler when speaking_until first crosses now).
    speaking_started_at: float = 0.0
    # Monotonic counter bumped on every barge-in. The turn handler captures it at
    # turn start; any outbound audio chunk whose captured value no longer matches
    # is a superseded chunk and is dropped, so a turn losing the cancel race
    # can't send audio after the flush or re-advance speaking_until.
    barge_generation: int = 0
    # True only while the opening greeting turn runs, so line noise at call start
    # can't cancel the greeting before the caller has spoken (the greeting is
    # still interruptible once it is audibly playing).
    greeting_active: bool = False
    # Serializes outbound transport writes (audio sends and the barge-in flush)
    # so the inbound task's clear() can't interleave with an in-flight send on
    # the same WebSocket — Starlette is not safe for two concurrent senders.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
