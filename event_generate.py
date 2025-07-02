#!/usr/bin/env python3
"""
generate_events.py
------------------
Use a local Gemma model to generate N fundraising events and save as JSON.

Example:
    python generate_events.py \\
           --num_events 50 \\
           --out_file  data/events_list.json
"""

import argparse, json, sys, time
from typing import List, Dict, Any

from transformers import pipeline

_GENERATOR = None


# ---------- 你可以在這裡增刪欄位 ----------
JSON_SCHEMA = {
    "event_id": "string (UUID)",
    "title": "string",
    "description": "string",
    "format": "string  # e.g. Gala Dinner / Charity Run / Webinar",
    "venue": {
        "name": "string",
        "city": "string",
        "lat": "float",
        "lon": "float"
    },
    "date_start": "ISO-8601 datetime",
    "date_end": "ISO-8601 datetime",
    "cause": "string",
    "sub_cause": "string",
    "keywords": ["string"],
    "target_segments": ["string"],
    "religious_affinity": "string or null",
    "age_range": ["int(min), int(max)"],
    "language": "string  # e.g. en / zh / zh-HK",
    "goal_amount": "int  # in USD",
    "donation_tiers": [
        {"label": "string", "min": "int"}
    ],
    "volunteer_slots": "int",
    "comm_channels": ["string  # e.g. Email, SMS"],
    "send_schedule": {
        "save_the_date": "date",
        "reminder": "date"
    },
    "prev_years": [
        {"year": "int", "attendees": "int", "total_raised": "int"}
    ],
    "sponsor_names": ["string"],
    "matching_ratio": "float",
    "tax_deductible": "boolean"
}
# ----------------------------------------


def build_prompt(num_events: int) -> str:
    """產生給 LLM 的提示詞 (英文較穩定)"""
    schema_str = json.dumps(JSON_SCHEMA, indent=2)
    return (
        f"You are a fundraising event planner.\n"
        f"Generate a JSON array containing exactly {num_events} different fundraising events. "
        f"Each element must strictly follow the following JSON schema (no extra keys, no comments):\n"
        f"{schema_str}\n\n"
        f"Rules:\n"
        f"1. Use realistic data; vary cause, format, target_segments, language.\n"
        f"2. Use ISO-8601 for all datetimes, UTC+08:00 timezone preferred.\n"
        f"3. Generate unique UUIDv4 for event_id.\n"
        f"4. Output ONLY valid JSON (no markdown, no code fences, no explanation).\n"
    )


def call_llm(prompt: str, timeout: int = 90) -> str:
    result = _GENERATOR(prompt, max_new_tokens=2048, do_sample=False)
    text = result[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def try_generate(n: int, retries: int = 3, delay: int = 5) -> List[Dict[str, Any]]:
    prompt = build_prompt(n)
    for attempt in range(1, retries + 1):
        try:
            raw = call_llm(prompt)
            data = json.loads(raw)
            if isinstance(data, list) and len(data) == n:
                return data
            raise ValueError(f"Returned JSON is not a list[{n}]")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Failed to generate valid JSON after retries")


def main():
    ap = argparse.ArgumentParser(description="Generate fundraising events via a local Gemma model")
    ap.add_argument("--num_events", type=int, default=50, help="Number of events to generate")
    ap.add_argument("--out_file", default="events_list.json", help="Output JSON file")
    ap.add_argument("--model", default="gemma-4b", help="HF model name or path")
    args = ap.parse_args()

    global _GENERATOR
    _GENERATOR = pipeline("text-generation", model=args.model)

    try:
        events = try_generate(args.num_events)
    except Exception as e:
        print(f"❌  Generation failed: {e}")
        sys.exit(1)

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"✅  Generated {len(events)} events → {args.out_file}")


if __name__ == "__main__":
    main()
