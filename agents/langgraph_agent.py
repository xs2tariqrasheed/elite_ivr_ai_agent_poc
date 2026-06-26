"""Reusable VoiceAgent built on a LangGraph ReAct agent.

Concrete agents supply a system prompt, tools, and an optional state snapshot;
this class handles invocation, memory, and reply rollback. Subclass it or
construct it directly from an agent's `build` factory.
"""
import logging
from typing import AsyncIterator, Callable, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
)
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

    async def checkpoint(self) -> Optional[set]:
        try:
            state = await self._agent.aget_state(self._config)
            return {
                m.id
                for m in state.values.get("messages", [])
                if getattr(m, "id", None)
            }
        except Exception:  # noqa: BLE001
            # Return None (not an empty set) so a lookup failure routes
            # rollback_barge to its safe fallback rather than the
            # remove-everything-not-in-pre_ids branch (an empty set would match
            # nothing, wiping the whole conversation).
            return None

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

    async def rollback_barge(
        self, pre_ids: Optional[set] = None, *, keep_user_message: bool = False
    ) -> None:
        """Restore memory to the exact pre-turn checkpoint after a barge-in.

        Unlike `rollback` (which drops only the last spoken reply and keeps the
        user message), this removes EVERY message added by the interrupted turn —
        the abandoned assistant reply, the caller utterance that triggered it,
        and any tool-call/ToolMessage pairs from the turn — so the caller's
        redo starts from a clean slate (the user's "prune and redo" requirement).

        It removes all trailing messages whose id is not in `pre_ids`, walking
        contiguously from the tail back to the first id present in `pre_ids`.
        Because the walk is contiguous, an AIMessage carrying tool_calls and its
        matching ToolMessage(s) are always removed together — never leaving an
        orphaned tool_call (which would 400 the next OpenAI request). Tool side
        effects already executed (e.g. a saved reservation) are NOT undone.

        With `keep_user_message=True` the caller's utterance is PRESERVED. Use
        this when the turn was interrupted before the agent produced any spoken
        reply: the caller did not interrupt a reply, they simply kept talking
        (e.g. STT split one sentence into fragments on a pause, and each fragment
        started a turn whose gap filler the continued speech "barged"). Discarding
        those fragments is what made the agent lose the pickup address and confuse
        it with the drop-off. Only this turn's assistant/tool messages (an
        abandoned partial reply, or an orphaned tool-call pair) are removed; the
        HumanMessage(s) survive so the fragment carries into the next turn.
        """
        if keep_user_message:
            await self._rollback_keep_user(pre_ids)
            return
        if not pre_ids:
            # No usable checkpoint (None, or empty — e.g. the opening greeting,
            # whose pre-turn memory was empty). We can't tell this turn's
            # messages apart from prior ones, and removing "everything not in an
            # empty set" would wipe the whole conversation — so fall back to
            # dropping just the last spoken reply.
            await self.rollback(None)
            return
        try:
            state = await self._agent.aget_state(self._config)
            to_remove = []
            for m in reversed(state.values.get("messages", [])):
                mid = getattr(m, "id", None)
                if mid is None:
                    # Can't address it for removal; stop to stay contiguous.
                    break
                if mid in pre_ids:
                    break
                to_remove.append(mid)
            if to_remove:
                await self._agent.aupdate_state(
                    self._config,
                    {"messages": [RemoveMessage(id=mid) for mid in to_remove]},
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not roll back interrupted turn: %s", exc)

    async def _rollback_keep_user(self, pre_ids: Optional[set]) -> None:
        """Remove an interrupted turn's reply/tool messages but keep its utterance.

        Walks contiguously from the tail, removing trailing messages until it
        reaches a HumanMessage (this turn's caller utterance) or a message from a
        prior turn (`pre_ids`), at which point it stops — so the caller's words
        are preserved and earlier turns are never touched. Stopping at the first
        HumanMessage also keeps the walk contiguous, so a removed ToolMessage is
        always paired with its tool-call AIMessage (no orphan that would 400 the
        next request). In the common case (interrupted during the gap filler with
        no reply yet) only the lone HumanMessage exists, so nothing is removed.
        """
        try:
            state = await self._agent.aget_state(self._config)
            to_remove = []
            for m in reversed(state.values.get("messages", [])):
                mid = getattr(m, "id", None)
                if mid is None:
                    break
                if pre_ids and mid in pre_ids:
                    break
                if isinstance(m, HumanMessage):
                    # The caller's utterance for this turn — keep it, and stop so
                    # we never reach into earlier turns.
                    break
                to_remove.append(mid)
            if to_remove:
                await self._agent.aupdate_state(
                    self._config,
                    {"messages": [RemoveMessage(id=mid) for mid in to_remove]},
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not roll back interrupted turn (keep-user): %s", exc)
