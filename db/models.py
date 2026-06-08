"""SQLAlchemy ORM models for the IVR app."""
from datetime import datetime, date, time
from typing import List, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Time, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


class Account(Base):
    """An account / customer that can place reservations."""

    __tablename__ = "accounts_table"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    cid: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)

    reservations: Mapped[List["Reservation"]] = relationship(
        "Reservation",
        back_populates="account",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Account id={self.id} account_number={self.account_number!r} name={self.name!r}>"


class Reservation(Base):
    """A pickup / drop-off reservation tied to an account."""

    __tablename__ = "reservations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts_table.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # reservation_number: Mapped[Optional[str]] = mapped_column(
    #     String(64), nullable=True, index=True
    # )
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    pickup_date: Mapped[date] = mapped_column(Date, nullable=False)
    pickup_time: Mapped[time] = mapped_column(Time, nullable=False)
    pickup_address: Mapped[str] = mapped_column(String(512), nullable=False)
    drop_off_address: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    account: Mapped["Account"] = relationship("Account", back_populates="reservations")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Reservation id={self.id} account_id={self.account_id} "
            f"{self.first_name} {self.last_name} {self.pickup_date} {self.pickup_time}>"
        )
