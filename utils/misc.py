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
