"""Order state for a single customer connection."""
from typing import Optional

from agents.order.menu import DRINKS, PIZZA_SIZES, SIDES, TOPPING_PRICE


class OrderSession:
    """Holds the mutable order state for one browser session."""

    def __init__(self) -> None:
        self.pizzas: list[dict] = []
        self.sides: list[dict] = []
        self.drinks: list[dict] = []
        self.customer_name: Optional[str] = None
        self.finalized: bool = False

    def subtotal(self) -> float:
        total = 0.0
        for p in self.pizzas:
            base = PIZZA_SIZES.get(p["size"], 0.0)
            total += (base + TOPPING_PRICE * len(p["toppings"])) * p["quantity"]
        for s in self.sides:
            total += SIDES.get(s["name"], 0.0) * s["quantity"]
        for d in self.drinks:
            total += DRINKS.get(d["name"], 0.0) * d["quantity"]
        return round(total, 2)

    def to_dict(self) -> dict:
        return {
            "customer_name": self.customer_name,
            "pizzas": self.pizzas,
            "sides": self.sides,
            "drinks": self.drinks,
            "subtotal": self.subtotal(),
            "finalized": self.finalized,
        }
