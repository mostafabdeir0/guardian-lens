"""Minimal Gemini access check. Does not run the research experiment."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai


MODEL_ID = "gemini-3-flash-preview"


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "replace_with_your_private_key":
        print("ERROR: GEMINI_API_KEY is not configured in your local .env file.")
        return 1

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents='Return only this JSON object: {"status":"ok"}',
    )

    text = (response.text or "").strip()

    print(f"Model configured: {MODEL_ID}")
    print(f"Response: {text}")

    return 0 if '"status"' in text and '"ok"' in text else 2


if __name__ == "__main__":
    sys.exit(main())