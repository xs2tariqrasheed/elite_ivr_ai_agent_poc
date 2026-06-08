"""Reservation state for a single existing-customer call."""
import random
import string
from typing import Optional


class ReservationSession:
    """Holds the mutable reservation state for one browser session.

    Seeded with the existing customer's account so the agent knows who is
    calling and where to mail the confirmation.
    """

    def __init__(self, account: Optional[dict] = None) -> None:
        account = account or {}
        self.caller_name: Optional[str] = account.get("name")
        self.caller_email: Optional[str] = account.get("email")
        self.caller_phone: Optional[str] = account.get("phone")

        self.pickup_datetime: Optional[str] = None
        self.pickup_address: Optional[str] = None
        self.dropoff_address: Optional[str] = None

        self.confirmed: bool = False
        self.confirmation_number: Optional[str] = None
        self.transferred: bool = False

    def generate_confirmation_number(self) -> str:
        """Assign and return a confirmation number like 'AJX123'."""
        letters = "".join(random.choices(string.ascii_uppercase, k=3))
        digits = "".join(random.choices(string.digits, k=3))
        self.confirmation_number = f"{letters}{digits}"
        return self.confirmation_number

    def to_dict(self) -> dict:
        return {
            "caller_name": self.caller_name,
            "caller_email": self.caller_email,
            "caller_phone": self.caller_phone,
            "pickup_datetime": self.pickup_datetime,
            "pickup_address": self.pickup_address,
            "dropoff_address": self.dropoff_address,
            "confirmed": self.confirmed,
            "confirmation_number": self.confirmation_number,
            "transferred": self.transferred,
        }
