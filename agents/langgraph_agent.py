"""Reusable VoiceAgent built on a LangGraph ReAct agent.

Concrete agents supply a system prompt, tools, and an optional state snapshot;
this class handles invocation, memory, and reply rollback. Subclass it or
construct it directly from an agent's `build` factory.
"""
import logging
from typing import AsyncIterator, Callable, Optional, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

log = logging.getLogger("voice")


class LangGraphAgent:
    """A VoiceAgent backed by `create_react_agent` with in-memory conversation state."""

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: Sequence[BaseTool],
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        thread_id: str = "default",
        snapshot_fn: Optional[Callable[[], dict]] = None,
        opening_trigger: Optional[str] = None,
    ) -> None:
        # If set, the pipeline runs one synthetic turn with this text at the
        # start of the call so the agent can speak first (e.g. a greeting).
        self.opening_trigger = opening_trigger
        # Fail fast on a stalled LLM request: on a phone call a hung connection
        # would otherwise freeze the (serialized) turn for the client's default
        # of ~30s+. A tight per-request timeout with a couple of quick retries
        # bounds the worst case to a few seconds.
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            timeout=8,
            max_retries=2,
        )
        self._agent = create_react_agent(
            llm,
            list(tools),
            state_modifier=system_prompt,
            checkpointer=MemorySaver(),
        )
        self._config = {"configurable": {"thread_id": thread_id}}
        self._snapshot_fn = snapshot_fn

    async def respond(self, text: str) -> str:
        result = await self._agent.ainvoke(
            {"messages": [("user", text)]}, self._config
        )
        return result["messages"][-1].content or ""

    async def stream_response(self, text: str) -> AsyncIterator[str]:
        """Yield the spoken reply token-by-token as the agent produces it.

        Only the final assistant message is streamed; tool-calling steps emit
        empty content (their tokens go to tool_call_chunks) and are skipped, so
        tool side effects still run but their arguments are never spoken.
        """
        async for chunk, _meta in self._agent.astream(
            {"messages": [("user", text)]},
            self._config,
            stream_mode="messages",
        ):
            if (
                isinstance(chunk, AIMessageChunk)
                and not chunk.tool_calls
                and not chunk.tool_call_chunks
                and isinstance(chunk.content, str)
                and chunk.content
            ):
                yield chunk.content

    def snapshot(self) -> Optional[dict]:
        return self._snapshot_fn() if self._snapshot_fn else None

    async def checkpoint(self) -> set:
        try:
            state = await self._agent.aget_state(self._config)
            return {
                m.id
                for m in state.values.get("messages", [])
                if getattr(m, "id", None)
            }
        except Exception:  # noqa: BLE001
            return set()

    async def rollback(self, pre_ids: Optional[set] = None) -> None:
        """Drop the agent's last spoken reply from memory.

        Tool messages are kept — their side effects already happened. A reply
        present in `pre_ids` (i.e. from a prior turn) is left untouched.
        """
        try:
            state = await self._agent.aget_state(self._config)
            for m in reversed(state.values.get("messages", [])):
                if not (
                    isinstance(m, AIMessage)
                    and not m.tool_calls
                    and getattr(m, "id", None)
                ):
                    continue
                if pre_ids is not None and m.id in pre_ids:
                    break
                await self._agent.aupdate_state(
                    self._config, {"messages": [RemoveMessage(id=m.id)]}
                )
                break
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not roll back reply: %s", exc)
