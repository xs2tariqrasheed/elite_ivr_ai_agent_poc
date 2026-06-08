"""Tony's Pizza order-taking agent."""
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from agents.langgraph_agent import LangGraphAgent
from agents.order.menu import (
    AVAILABLE_TOPPINGS,
    DRINKS,
    PIZZA_SIZES,
    SIDES,
    format_menu,
    normalize,
)
from agents.order.session import OrderSession
from configs.settings import Settings

SYSTEM_PROMPT = """You are Tony, a friendly phone agent for "Tony's Pizza".
Your job is to take the customer's order by voice.

MENU:
{menu}

RULES:
- Your replies are read aloud by a text-to-speech engine, so keep them SHORT,
  natural, and spoken. One or two sentences. No markdown, no bullet points,
  no emojis, no prices listed unless asked.
- Always use the tools to record items. Never invent items not on the menu.
- A pizza needs a size (small, medium, large). If the customer doesn't say a
  size, ask for it before adding. Cheese is included; ask if they want toppings.
- Confirm each item briefly after adding it.
- If the customer asks for the total or to check the order, use get_order.
- When the customer says they're done, ask for their name if you don't have it,
  then read back the order and total and call finalize_order.
- Be warm and efficient. Greet the customer on the first turn.
"""


def _make_tools(session: OrderSession) -> List[BaseTool]:
    """Build the order tools bound to `session`."""

    @tool
    def add_pizza(
        size: str, toppings: Optional[List[str]] = None, quantity: int = 1
    ) -> str:
        """Add a pizza to the order. size must be small, medium, or large.
        toppings is a list of topping names (optional; cheese is always included)."""
        size = normalize(size)
        if size not in PIZZA_SIZES:
            return f"Invalid size '{size}'. Choose small, medium, or large."
        clean: list[str] = []
        rejected: list[str] = []
        for t in toppings or []:
            tn = normalize(t)
            if tn in AVAILABLE_TOPPINGS:
                clean.append(tn)
            else:
                rejected.append(t)
        quantity = max(1, int(quantity))
        session.pizzas.append({"size": size, "toppings": clean, "quantity": quantity})
        msg = f"Added {quantity} {size} pizza(s)"
        if clean:
            msg += " with " + ", ".join(clean)
        if rejected:
            msg += f". Note: not available: {', '.join(rejected)}"
        return msg

    @tool
    def add_side(name: str, quantity: int = 1) -> str:
        """Add a side item (e.g. garlic bread, chicken wings) to the order."""
        n = normalize(name)
        if n not in SIDES:
            return f"'{name}' is not on the sides menu. Available: {', '.join(SIDES)}"
        session.sides.append({"name": n, "quantity": max(1, int(quantity))})
        return f"Added {quantity} {n}"

    @tool
    def add_drink(name: str, quantity: int = 1) -> str:
        """Add a drink (e.g. coke, sprite, water) to the order."""
        n = normalize(name)
        if n not in DRINKS:
            return f"'{name}' is not on the drinks menu. Available: {', '.join(DRINKS)}"
        session.drinks.append({"name": n, "quantity": max(1, int(quantity))})
        return f"Added {quantity} {n}"

    @tool
    def remove_item(name: str) -> str:
        """Remove the most recently added item matching a name or size keyword."""
        key = normalize(name)
        for bucket in (session.pizzas, session.sides, session.drinks):
            for i in range(len(bucket) - 1, -1, -1):
                item = bucket[i]
                hay = " ".join(str(v) for v in item.values())
                if key in hay:
                    bucket.pop(i)
                    return f"Removed {name}"
        return f"Couldn't find '{name}' in the order."

    @tool
    def set_customer_name(name: str) -> str:
        """Record the customer's name for the order."""
        session.customer_name = name.strip().title()
        return f"Name set to {session.customer_name}"

    @tool
    def get_order() -> str:
        """Return the current order contents and subtotal."""
        d = session.to_dict()
        if not (d["pizzas"] or d["sides"] or d["drinks"]):
            return "The order is currently empty."
        parts = []
        for p in d["pizzas"]:
            t = (" with " + ", ".join(p["toppings"])) if p["toppings"] else ""
            parts.append(f"{p['quantity']} {p['size']} pizza{t}")
        for s in d["sides"]:
            parts.append(f"{s['quantity']} {s['name']}")
        for dr in d["drinks"]:
            parts.append(f"{dr['quantity']} {dr['name']}")
        return "; ".join(parts) + f". Subtotal ${d['subtotal']:.2f}."

    @tool
    def finalize_order() -> str:
        """Finalize and confirm the order. Call when the customer is done."""
        if not (session.pizzas or session.sides or session.drinks):
            return "Cannot finalize an empty order."
        session.finalized = True
        return f"Order finalized. Total ${session.subtotal():.2f}."

    return [
        add_pizza, add_side, add_drink, remove_item,
        set_customer_name, get_order, finalize_order,
    ]


def build(settings: Settings, params: Optional[dict] = None) -> LangGraphAgent:
    """Create a fresh order agent for one connection."""
    session = OrderSession()
    return LangGraphAgent(
        system_prompt=SYSTEM_PROMPT.format(menu=format_menu()),
        tools=_make_tools(session),
        model=settings.openai_model,
        thread_id="order",
        snapshot_fn=session.to_dict,
    )
