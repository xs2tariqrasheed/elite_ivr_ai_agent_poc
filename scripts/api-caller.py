"""Call the account-name-audio API from the command line.

Usage:
    python scripts/api-caller.py
    python scripts/api-caller.py --account-name "Hi John Smith" --account-number 12345
    python scripts/api-caller.py --url http://localhost:8000/account-name-audio
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

GREETING = (
    "[warmly][cheerful] Thanks for calling Elite Limousine and Welcome back. "
    "[friendly] My name is Ann. "
    "[curious] Do you want a new reservation, or have questions about something else?"
)


DEFAULT_URL = "http://localhost:8000/account-name-audio"
DEFAULT_ACCOUNT_NUMBER = "known_greet"

message_to_say = GREETING

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the account-name-audio API endpoint."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL to call.")
    parser.add_argument(
        "--account-name",
        default=message_to_say,
        help="Account name text payload.",
    )
    parser.add_argument(
        "--account-number",
        default=DEFAULT_ACCOUNT_NUMBER,
        help="Account number payload.",
    )
    parser.add_argument(
        "--timeout", type=int, default=20, help="Request timeout in seconds."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "account_name": args.account_name,
        "account_number": args.account_number,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_text = response.read().decode("utf-8")
            print(f"Status: {response.status}")
            print(response_text)
            return 0
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"HTTPError: {error.code}")
        print(error_body)
        return 1
    except urllib.error.URLError as error:
        print(f"URLError: {error.reason}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
