"""Format-agnostic phone number matching.

Callers reach the IVR from any country and Twilio hands us E.164 (``+923234251430``),
while an admin may have typed the same line as ``03234251430`` or ``+92-323-4251430``.
We treat all of those as the same number by comparing only their trailing
*significant* digits, which discards country codes, national trunk prefixes,
brackets, spaces and dashes.
"""
import re
from typing import Optional

_NON_DIGITS = re.compile(r"\D+")

# Compare the last N digits. A national subscriber number is ~10 digits in the
# regions we serve; the leading country code (``92``) or trunk prefix (``0``)
# sits in front of it, so the trailing 10 digits identify the line.
_SIGNIFICANT = 10

# Below this many digits a trailing comparison is too loose (it could match an
# extension or a fragment), so we fall back to exact-digit equality.
_MIN_SUFFIX = 7


def phone_digits(value: Optional[str]) -> str:
    """Return only the digit characters of `value` (``"+92-323"`` -> ``"92323"``)."""
    return _NON_DIGITS.sub("", value or "")


def phones_match(a: Optional[str], b: Optional[str]) -> bool:
    """Return True if `a` and `b` are the same phone number, ignoring formatting.

    ``03234251430``, ``+923234251430`` and ``+92-323-4251430`` all match.
    """
    da, db = phone_digits(a), phone_digits(b)
    if not da or not db:
        return False
    n = min(_SIGNIFICANT, len(da), len(db))
    if n < _MIN_SUFFIX:
        return da == db
    return da[-n:] == db[-n:]
