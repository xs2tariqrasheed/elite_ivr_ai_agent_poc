"""Tony's Pizza menu data and helpers — used by the order agent only."""

PIZZA_SIZES: dict[str, float] = {
    "small": 9.0,
    "medium": 12.0,
    "large": 15.0,
}

# Extra toppings beyond cheese cost this per topping.
TOPPING_PRICE: float = 1.5

AVAILABLE_TOPPINGS: list[str] = [
    "pepperoni", "mushroom", "onion", "sausage", "bacon",
    "extra cheese", "black olives", "green peppers", "pineapple",
    "ham", "spinach", "jalapenos", "tomato", "chicken",
]

SIDES: dict[str, float] = {
    "garlic bread": 4.5,
    "cheesy bread": 6.0,
    "chicken wings": 8.0,
    "caesar salad": 6.5,
    "mozzarella sticks": 5.5,
}

DRINKS: dict[str, float] = {
    "coke": 2.5,
    "diet coke": 2.5,
    "sprite": 2.5,
    "water": 1.5,
    "lemonade": 3.0,
}


def format_menu() -> str:
    """Return a human-readable menu string for injection into the system prompt."""
    sizes = ", ".join(f"{s} (${p:.2f})" for s, p in PIZZA_SIZES.items())
    tops = ", ".join(AVAILABLE_TOPPINGS)
    sides = ", ".join(f"{s} (${p:.2f})" for s, p in SIDES.items())
    drinks = ", ".join(f"{d} (${p:.2f})" for d, p in DRINKS.items())
    return (
        f"PIZZAS — sizes: {sizes}. Each pizza includes cheese; "
        f"extra toppings are ${TOPPING_PRICE:.2f} each.\n"
        f"TOPPINGS: {tops}\n"
        f"SIDES: {sides}\n"
        f"DRINKS: {drinks}"
    )


def normalize(name: str) -> str:
    """Lowercase and strip a menu item name for consistent comparisons."""
    return (name or "").strip().lower()
