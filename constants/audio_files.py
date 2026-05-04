"""Constants for the pre-recorded agent audio files.

Each constant maps to a filename inside ``AUDIO_DIR`` (without the
``.mp3`` extension). The agent voice service uses these constants as
the lookup keys for the in-memory audio cache.
"""

GREET_UNKNOWN = "rec_greet_unknown"
OTHER_INTENT = "rec_other_intent"

ACCOUNT_NUMBER = "rec_account_number"
ACCOUNT_NUMBER_RETRY = "rec_account_number_retry"
ACCOUNT_NOT_FOUND = "rec_account_not_found"

ACCOUNT_NAME = "rec_account_name"
FIRST_NAME = "rec_first_name"
LAST_NAME = "rec_last_name"
PICKUP_DATE_TIME = "rec_pickup_date_time"
PICKUP_ADDRESS = "rec_pickup_address"
DROPOFF_ADDRESS = "rec_dropoff_address"
CALLBACK_NUMBER = "rec_callback_number"
EMAIL = "rec_email"

GOOD_BYE = "rec_good_bye"


# Convenient list of every audio file the app expects to load at startup.
ALL_AUDIO_FILES = [
    GREET_UNKNOWN,
    OTHER_INTENT,
    ACCOUNT_NUMBER,
    ACCOUNT_NUMBER_RETRY,
    ACCOUNT_NOT_FOUND,
    ACCOUNT_NAME,
    FIRST_NAME,
    LAST_NAME,
    PICKUP_DATE_TIME,
    PICKUP_ADDRESS,
    DROPOFF_ADDRESS,
    CALLBACK_NUMBER,
    EMAIL,
    GOOD_BYE,
]
