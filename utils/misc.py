import re
from typing import Optional


_YES_PATTERN = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|sure|affirmative|"
    r"absolutely|of course|that's right|thats right|ok|okay|confirm|confirmed)\b",
    re.IGNORECASE,
)
_NO_PATTERN = re.compile(
    r"\b(no|nope|nah|negative|incorrect|wrong|"
    r"that's wrong|thats wrong|not right|don't|dont)\b",
    re.IGNORECASE,
)


def detect_yes_no(text: str) -> Optional[bool]:
    """Classify a short caller response as yes/no.

    Returns True for a "yes" match, False for a "no" match, and None when
    neither pattern fires. "No" is checked first so phrases like
    "no, that's right" are classified as denial rather than confirmation.
    """
    if not text:
        return None
    if _NO_PATTERN.search(text):
        return False
    if _YES_PATTERN.search(text):
        return True
    return None


def is_valid_account_number(account_number: str) -> bool:
    """
    Check if the account number is valid.
    """
    if not account_number:
        return False
    if len(account_number) != 4:
        return False
    if not account_number.isdigit():
        return False
    return True


def is_valid_phone(phone: str) -> bool:
    """
    Check if the phone number is valid.
    """
    if not phone:
        return False
    if len(phone) != 10:
        return False
    if not phone.isdigit():
        return False
    return True
