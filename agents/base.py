"""The contract every pluggable agent must satisfy.

The voice pipeline (STT -> agent -> TTS) depends only on this protocol, so any
agent — order taking, booking, inquiry — can be dropped in without touching the
pipeline. See `agents.langgraph_agent.LangGraphAgent` for a ready-made base.
"""
from typing import AsyncIterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class VoiceAgent(Protocol):
    """A conversational agent driven turn-by-turn by the voice pipeline."""

    async def respond(self, text: str) -> str:
        """Process one user utterance and return the reply to be spoken."""
        ...

    def stream_response(self, text: str) -> AsyncIterator[str]:
        """Yield the spoken reply incrementally so TTS can start before it ends."""
        ...

    def snapshot(self) -> Optional[dict]:
        """Structured state to push to the client, or None if the agent has none.

        Domain-specific (e.g. the current order); the client decides how to render it.
        """
        ...

    async def checkpoint(self) -> set:
        """Return an opaque marker of the conversation state before a turn.

        Passed back to `rollback` so only the new reply is dropped.
        """
        ...

    async def rollback(self, pre_ids: Optional[set] = None) -> None:
        """Drop the last spoken reply from memory.

        Used when a turn is abandoned (session teardown) or produced no usable
        reply, so the unanswered message doesn't linger in agent memory.
        `pre_ids` is a checkpoint from before the turn; a reply predating it is
        left alone. When None, the most recent reply is dropped unconditionally.
        """
        ...
