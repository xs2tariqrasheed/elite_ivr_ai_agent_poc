# accounts

ACCOUNT_1 = {
    "name": "Tariq Rasheed",
    "phone": "+923234251430",
    "email": "xs2tariqrasheed@gmail.com"
}

ACCOUNT_2 = {
    "name": "John Smith",
    "phone": "+923004433123",
    "email": "john.smith@xyz.com"
}

# Existing customers, indexed by position. The frontend picks one before a call
# so the reservation agent knows who is calling.
ACCOUNTS = [ACCOUNT_1, ACCOUNT_2]


def get_account(index: int):
    """Return the account at `index`, or None if out of range."""
    if 0 <= index < len(ACCOUNTS):
        return ACCOUNTS[index]
    return None


def get_account_by_phone(phone: str):
    """Return the account whose phone matches `phone` (E.164), or None."""
    for account in ACCOUNTS:
        if account["phone"] == phone:
            return account
    return None
