"""Maps an agent name to its builder so the pipeline can stay agent-agnostic.

To add a new agent: implement a `build(settings, params=None) -> VoiceAgent`
factory in its own package under `agents/`, then register it here.
"""
from typing import Callable, Dict, Optional

from agents.base import VoiceAgent
from agents.order import agent as order_agent
from agents.reservation import agent as reservation_agent
from configs.settings import Settings

# name -> factory. Each factory returns a fresh agent for one connection.
# `params` carries per-connection context (e.g. the selected caller account).
_BUILDERS: Dict[str, Callable[[Settings, Optional[dict]], VoiceAgent]] = {
    "order": order_agent.build,
    "reservation": reservation_agent.build,
}


def available_agents() -> list[str]:
    return sorted(_BUILDERS)


def build_agent(
    name: str, settings: Settings, params: Optional[dict] = None
) -> VoiceAgent:
    """Build the agent registered under `name`."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown agent '{name}'. Available: {', '.join(available_agents())}"
        )
    return builder(settings, params)
