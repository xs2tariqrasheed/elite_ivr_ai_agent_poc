"""Tiny helper to look up accounts in the dummy data file."""
import json
import logging
import os
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


_accounts: List[Dict] = []


def load_accounts() -> None:
    """Read ``DUMMY_DATA_PATH`` once at startup."""
    global _accounts
    path = config.DUMMY_DATA_PATH
    if not os.path.exists(path):
        logger.warning("Dummy data file not found at %s — accounts list will be empty", path)
        _accounts = []
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            _accounts = json.load(f)
        logger.info("Loaded %d dummy accounts from %s", len(_accounts), path)
    except Exception:
        logger.exception("Failed to load dummy accounts from %s", path)
        _accounts = []


def find_account_by_number(account_number: str) -> Optional[Dict]:
    """Return the account dict for ``account_number`` or ``None``."""
    if not account_number:
        return None
    for acc in _accounts:
        if str(acc.get("account_number")) == str(account_number):
            return acc
    return None
