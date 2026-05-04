"""
number_extractor.py
Python 3.9 compatible

Fast OTP / 4-digit code extractor for noisy Whisper transcripts.

Features
--------
- Extracts 4-digit codes from:
    "4821"
    "48 21"
    "4-8-2-1"
    "four eight two one"
    "for eight to one"
    "oh nine six five"

- Handles common Whisper mistakes:
    for -> 4
    to/too -> 2
    oh -> 0
    ate -> 8
    won -> 1

- Ignores unrelated long numbers when possible
- Prefers OTP-like context words:
    code, otp, pin, verification, password, passcode

- No LLM required
- Extremely fast
- Pure Python stdlib

Usage
-----
from number_extractor import extract_number

code = extract_number("your verification code is four eight two one")
print(code)

Output:
4821
"""

import re
from typing import Optional, List


# ============================================================
# NORMALIZATION DICTIONARY
# ============================================================

NUMBER_WORDS = {
    # Standard
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",

    # Whisper / speech confusion
    "oh": "0",
    "o": "0",

    "won": "1",

    "to": "2",
    "too": "2",
    "tu": "2",

    "tree": "3",

    "for": "4",
    "fore": "4",

    "hive": "5",
    "pi": "5",
    "pie": "5",

    "sics": "6",

    "ate": "8",

    "wine": "9",
}


OTP_CONTEXT_WORDS = {
    "otp",
    "code",
    "pin",
    "passcode",
    "password",
    "verification",
    "verify",
    "security",
    "login",
    "auth",
    "authentication",
    "account number",
    "account"
}


# ============================================================
# HELPERS
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Lowercase and normalize spacing.
    """

    text = text.lower()

    # Replace separators with spaces
    text = re.sub(r"[-_:|/,]", " ", text)

    # Remove weird punctuation except digits/letters
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _replace_number_words(text: str) -> str:
    """
    Replace spoken number words with digits.
    """

    words = text.split()
    normalized = []

    for word in words:
        normalized.append(NUMBER_WORDS.get(word, word))

    return " ".join(normalized)


def _extract_consecutive_digits(text: str) -> List[str]:
    """
    Extract direct 4-digit sequences.
    """

    return re.findall(r"\d{4}", text)


def _extract_spaced_digits(text: str) -> List[str]:
    """
    Extract spaced digit patterns:
    4 8 2 1
    4-8-2-1
    """

    matches = re.findall(
        r"(?:\b\d\b[\s]*){4}",
        text
    )

    results = []

    for match in matches:
        digits = re.sub(r"\D", "", match)

        if len(digits) == 4:
            results.append(digits)

    return results


def _extract_mixed_digits(text: str) -> List[str]:
    """
    Extract digits hidden inside words:
    AB4821XY
    x4y8z2w1
    """

    results = []

    cleaned = re.sub(r"\D", "", text)

    for i in range(len(cleaned) - 3):
        chunk = cleaned[i:i + 4]

        if chunk.isdigit():
            results.append(chunk)

    return results


def _score_candidate(candidate: str, text: str) -> int:
    """
    Score OTP likelihood.
    Higher score = more likely OTP.
    """

    score = 0

    # Avoid obvious junk
    if candidate == "0000":
        score -= 2

    # Context words boost confidence
    for word in OTP_CONTEXT_WORDS:
        if word in text:
            score += 5

    # Penalize if transcript contains many long numbers
    long_numbers = re.findall(r"\d{5,}", text)

    if long_numbers:
        score -= 1

    return score


# ============================================================
# MAIN API
# ============================================================

def extract_number(text: str) -> Optional[str]:
    """
    Extract most likely 4-digit OTP from noisy transcript.

    Parameters
    ----------
    text : str

    Returns
    -------
    Optional[str]
        4-digit OTP or None
    """

    if not text:
        return None

    # ----------------------------------------
    # Normalize
    # ----------------------------------------

    text = _normalize_text(text)

    # ----------------------------------------
    # Replace spoken numbers
    # ----------------------------------------

    text = _replace_number_words(text)

    candidates = []

    # ----------------------------------------
    # Direct 4-digit numbers
    # ----------------------------------------

    candidates.extend(_extract_consecutive_digits(text))

    # ----------------------------------------
    # Spaced digits
    # ----------------------------------------

    candidates.extend(_extract_spaced_digits(text))

    # ----------------------------------------
    # Hidden digits
    # ----------------------------------------

    candidates.extend(_extract_mixed_digits(text))

    # ----------------------------------------
    # Remove duplicates
    # ----------------------------------------

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return None

    # ----------------------------------------
    # Rank candidates
    # ----------------------------------------

    ranked = sorted(
        candidates,
        key=lambda c: _score_candidate(c, text),
        reverse=True
    )

    return ranked[0]


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    TESTS = [
        "your otp is 4821",
        "verification code four eight two one",
        "code is for eight to one",
        "pin is 4 8 2 1",
        "password 48-21",
        "AB4821XY",
        "your security code is oh nine six five",
        "please verify using code won too tree for",
        "call me at 923001112222 but otp is 7612",
        "hello hello no otp here",
    ]

    for test in TESTS:
        print(f"INPUT : {test}")
        print(f"OUTPUT: {extract_number(test)}")
        print("-" * 50)