"""
phone_extractor.py
Python 3.9+

Fast phone number extractor for noisy speech-to-text input.

Extracts 10-digit phone numbers from:
    "9876543210"
    "nine eight seven six five four three two one zero"
    "98-76-54-32-10"
    "call me at nine eight seven six..."
    "my number is 98765 43210"

Handles common Whisper mistakes:
    for -> 4
    to/too -> 2
    oh/o -> 0
    won -> 1
    ate -> 8

Author: ChatGPT
"""

import re
from typing import Optional, List

# ------------------------------------------------------------
# Number mappings
# ------------------------------------------------------------

DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "tree": "3",
    "four": "4",
    "for": "4",
    "fore": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}

PHONE_CONTEXT = {
    "phone",
    "number",
    "mobile",
    "contact",
    "call",
    "cell",
    "telephone",
}

TOKEN_REGEX = re.compile(r"[a-zA-Z0-9]+")

PHONE_REGEX = re.compile(r"\d{10}")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    return TOKEN_REGEX.findall(text.lower())


def normalize_token(token: str) -> str:

    token = token.lower().strip()

    if token in DIGIT_WORDS:
        return DIGIT_WORDS[token]

    return token


def token_to_digits(token: str) -> str:

    token = normalize_token(token)

    # direct digit word
    if token in DIGIT_WORDS.values():
        return token

    # direct numeric
    if token.isdigit():
        return token

    # mixed token
    return re.sub(r"\D", "", token)


# ------------------------------------------------------------
# Main extractor
# ------------------------------------------------------------


def extract_phone(text: str) -> Optional[str]:

    if not text or not isinstance(text, str):
        return None

    # --------------------------------------------------------
    # STEP 1: direct extraction
    # --------------------------------------------------------

    direct_digits = re.sub(r"\D", "", text)

    match = PHONE_REGEX.search(direct_digits)

    if match:
        return match.group()

    # --------------------------------------------------------
    # STEP 2: tokenize
    # --------------------------------------------------------

    tokens = tokenize(text)

    # --------------------------------------------------------
    # STEP 3: context-based extraction
    # --------------------------------------------------------

    for i, token in enumerate(tokens):
        if token in PHONE_CONTEXT:
            collected = ""

            for future in tokens[i + 1 : i + 20]:
                collected += token_to_digits(future)

                if len(collected) >= 10:
                    match = PHONE_REGEX.search(collected)

                    if match:
                        return match.group()

    # --------------------------------------------------------
    # STEP 4: full reconstruction
    # --------------------------------------------------------

    reconstructed = ""

    for token in tokens:
        reconstructed += token_to_digits(token)

    match = PHONE_REGEX.search(reconstructed)

    if match:
        return match.group()

    return None


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------


def is_valid_phone(phone: str) -> bool:
    """
    Check if string is exactly a 10-digit number.
    """

    return bool(re.fullmatch(r"\d{10}", phone))


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

if __name__ == "__main__":
    TESTS = [
        # clean
        "9876543210",
        # spaced
        "98765 43210",
        # punctuation
        "98-76-54-32-10",
        # spoken
        "nine eight seven six five four three two one zero",
        # whisper mistakes
        "nine eight seven six five for three to one oh",
        # mixed
        "98 seven six 54 three two one 0",
        # sentence
        "my phone number is 9876543210",
        # noisy
        "hello sir my mobile number is nine eight seven six five four three two one zero thank you",
        # invalid
        "call me tomorrow",
    ]

    for text in TESTS:
        result = extract_phone(text)

        print(f"INPUT : {text}")
        print(f"OUTPUT: {result}")
        print("-" * 60)
