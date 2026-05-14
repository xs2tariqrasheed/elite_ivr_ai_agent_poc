"""Call the gen-audio API from the command line.

Usage:
    python scripts/api-tts.py
    python scripts/api-tts.py --text "Hi this is Ann. Welcome back to Elite Limousine." --file-name ann_welcome_back.mp3
    python scripts/api-tts.py --url http://localhost:8000/gen-audio
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

# GREETING = (
#     "[warmly][cheerful] Thanks for calling Elite Limousine and Welcome back. "
#     "[friendly] My name is Ann. "
#     "[curious] Do you want a new reservation, or have questions about something else?"
# )


DEFAULT_URL = "http://localhost:8000/gen-audio"

message_to_say = "Hi this is Ann. Welcome back to Elite Limousine."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the gen-audio API endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL to call.")
    parser.add_argument(
        "--text",
        default=message_to_say,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--file-name",
        default="ann_welcome_back.mp3",
        help="File name to save the audio to.",
    )
    parser.add_argument(
        "--timeout", type=int, default=20, help="Request timeout in seconds."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "text": args.text,
        "file_name": args.file_name,
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
