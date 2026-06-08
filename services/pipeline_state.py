"""Mutable state shared across the pipeline components for one connection."""
import asyncio
from dataclasses import dataclass


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
    # own voice can't bleed back in and corrupt / stall turn detection.
    speaking_until: float = 0.0
