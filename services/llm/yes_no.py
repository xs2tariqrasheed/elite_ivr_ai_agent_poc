"""Yes / no agreement detection from call transcripts using LLM."""
import logging
import re

import config

from ._helpers import _llm_generate

logger = logging.getLogger(__name__)

_openai_client = None


def _get_openai_client():
    """Lazily build and cache the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client

# Fallback when the model is unavailable or returns unusable text.
# NO patterns take precedence over YES when both appear (e.g. "yes I don't want that").
_NO_RE = re.compile(
    r"\b("
    r"no|nope|nah|naw|never|negative|wrong|incorrect|"
    r"not really|not at all|no way|no thanks|"
    r"don'?t|do not|can'?t|cannot|won'?t|shouldn'?t|"
    r"cancel"
    r")\b",
    re.I,
)
_YES_RE = re.compile(
    r"\b("
    r"yes|yeah|yep|yup|yah|ya\b|sure|ok|okay|okey|"
    r"correct|right|absolutely|definitely|indeed|exactly|"
    r"please|affirmative|fine|agreed|agree|"
    r"sounds good|go ahead|that works|that'?s fine|that'?s right|"
    r"uh-?huh|mm-?hm+|mhm|uhuh|hm\s*hm|hm+\s*hm+|"
    r"you bet|of course|for sure|by all means"
    r")\b",
    re.I,
)


def _keyword_agreement(text: str) -> bool:
    """Best-effort fallback from transcript keywords."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if _NO_RE.search(t):
        return False
    return bool(_YES_RE.search(t))


def detect_yes_no_llm(text: str) -> bool:
    """Decide if the caller's spoken reply is agreement (yes) or not.

    ``text`` is typically speech-to-text from the call. The model should
    treat informal affirmatives (e.g. "yeah", "yup", "mm-hmm", "uh-huh",
    "hm hm", "sure", "sounds good") as agreement, and negations or
    refusals as non-agreement. Unclear, off-topic, or hedging replies
    (e.g. "maybe", "I don't know") are treated as non-agreement.

    Returns ``True`` only when the reply clearly expresses agreement;
    ``False`` otherwise.
    """
    text = (text or "").strip()
    if not text:
        return False

    prompt = (
        "You are classifying a caller's short reply on a phone IVR. The "
        "text below is a speech-to-text transcript; it may be noisy, "
        "truncated, or contain homophone errors.\n\n"
        "Your task: did the caller EXPRESS AGREEMENT / CONFIRMATION / YES "
        "to what was asked (as in 'yes', 'I agree', 'that's correct', "
        "'go ahead', 'please do')?\n\n"
        "Treat ALL of the following as AGREEMENT (YES): yes, yeah, yep, yup, "
        "yah, sure, ok, okay, right, correct, absolutely, definitely, "
        "indeed, exactly, please, fine, agreed, 'sounds good', 'go ahead', "
        "'that works', 'that's fine', 'you bet', 'of course', 'for sure', "
        "affirmative, and spoken backchannels that mean 'yes' such as "
        "uh-huh, mm-hmm, mhm, hm-hm, hm hm, repeated 'hm' or 'mm' used as "
        "nodding along.\n\n"
        "Treat ALL of the following as NON-AGREEMENT (NO): no, nope, nah, "
        "never, negative, wrong, incorrect, 'not really', 'not at all', "
        "refusals, cancellations, 'I don't want', 'stop', 'wait', strong "
        "disagreement, or asking to change their mind.\n\n"
        "If the reply is ambiguous, off-topic, only partial words, pure "
        "hesitation without clear yes, or impossible to tell, answer NO.\n\n"
        "Respond with ONLY one token and nothing else: YES or NO.\n\n"
        f"Transcript: {text}\n"
        "Answer:"
    )
    try:
        raw = _llm_generate(prompt, num_predict=8)
    except Exception:
        logger.exception("Ollama call failed for yes/no detection")
        raw = ""

    logger.debug("LLM yes/no raw response: %r", raw)

    m = re.search(r"\b(YES|NO)\b", raw or "", re.I)
    if m:
        agreed = m.group(1).upper() == "YES"
    else:
        agreed = _keyword_agreement(text)

    logger.info("Yes/no from transcript: %s (from %r)", agreed, text)
    return agreed


def detect_yes_no_llm_openai(text: str) -> bool:
    """Decide if the caller's spoken reply is agreement (yes) or not.

    Same contract as :func:`detect_yes_no_llm` but routes the prompt
    through the OpenAI Chat Completions API instead of Ollama.
    """
    text = (text or "").strip()
    if not text:
        return False

    prompt = (
        "You are classifying a caller's short reply on a phone IVR. The "
        "text below is a speech-to-text transcript; it may be noisy, "
        "truncated, or contain homophone errors.\n\n"
        "Your task: did the caller EXPRESS AGREEMENT / CONFIRMATION / YES "
        "to what was asked (as in 'yes', 'I agree', 'that's correct', "
        "'go ahead', 'please do')?\n\n"
        "Treat ALL of the following as AGREEMENT (YES): yes, yeah, yep, yup, "
        "yah, sure, ok, okay, right, correct, absolutely, definitely, "
        "indeed, exactly, please, fine, agreed, 'sounds good', 'go ahead', "
        "'that works', 'that's fine', 'you bet', 'of course', 'for sure', "
        "affirmative, and spoken backchannels that mean 'yes' such as "
        "uh-huh, mm-hmm, mhm, hm-hm, hm hm, repeated 'hm' or 'mm' used as "
        "nodding along.\n\n"
        "Treat ALL of the following as NON-AGREEMENT (NO): no, nope, nah, "
        "never, negative, wrong, incorrect, 'not really', 'not at all', "
        "refusals, cancellations, 'I don't want', 'stop', 'wait', strong "
        "disagreement, or asking to change their mind.\n\n"
        "If the reply is ambiguous, off-topic, only partial words, pure "
        "hesitation without clear yes, or impossible to tell, answer NO.\n\n"
        "Respond with ONLY one token and nothing else: YES or NO.\n\n"
        f"Transcript: {text}\n"
        "Answer:"
    )

    raw = ""
    try:
        client = _get_openai_client()
        completion = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("OpenAI call failed for yes/no detection")

    logger.debug("OpenAI yes/no raw response: %r", raw)

    m = re.search(r"\b(YES|NO)\b", raw or "", re.I)
    if m:
        agreed = m.group(1).upper() == "YES"
    else:
        agreed = _keyword_agreement(text)

    logger.info("Yes/no from transcript (openai): %s (from %r)", agreed, text)
    return agreed
