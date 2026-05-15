"""Call flow phase constants.

The IVR walks through these phases in order. See ``call_flow.txt`` for
the original specification.
"""

# Phase identifiers (kept as plain strings so they can be logged/serialised
# easily without enum unwrapping).
PHASE_START = "start"
PHASE_INTENT = "phase_1_intent"
PHASE_ACCOUNT_NUMBER = "phase_2_account_number"
PHASE_ACCOUNT_NAME = "phase_3_account_name"
PHASE_FIRST_NAME = "phase_4_first_name"
PHASE_LAST_NAME = "phase_5_last_name"
PHASE_PICKUP_DATE_TIME = "phase_6_pickup_date_time"
PHASE_PICKUP_ADDRESS = "phase_7_pickup_address"
PHASE_DROPOFF_ADDRESS = "phase_8_dropoff_address"
PHASE_CALLBACK_NUMBER = "phase_9_callback_number"
PHASE_EMAIL = "phase_10_email"
PHASE_END = "phase_11_end"
PHASE_HANGUP = "hangup"

PHASE_PASSENGER_INFO_VERIFICATION = "phase_2_passenger_info_verification"


# Linear ordering of the happy-path phases. The phase manager uses this
# to drive the call from one phase to the next when a phase succeeds.
PHASE_ORDER = [
    PHASE_INTENT,
    PHASE_PASSENGER_INFO_VERIFICATION,
    PHASE_ACCOUNT_NUMBER,
    PHASE_ACCOUNT_NAME,
    PHASE_FIRST_NAME,
    PHASE_LAST_NAME,
    PHASE_PICKUP_DATE_TIME,
    PHASE_PICKUP_ADDRESS,
    PHASE_DROPOFF_ADDRESS,
    PHASE_CALLBACK_NUMBER,
    PHASE_EMAIL,
    PHASE_END,
]
