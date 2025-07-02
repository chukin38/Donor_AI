#!/usr/bin/env python3
"""
generate_events.py
------------------
以 ChatGPT API 生成 N 筆「捐贈者活動」並存成 JSON 檔

例：
    python generate_events.py \
           --num_events 50 \
           --out_file  data/events_list.json \
           --api_url   http://57.129.18.204:51001/generate-text
"""

import argparse, json, sys, time, uuid
from typing import List, Dict, Any

import requests


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


def call_llm(api_url: str, prompt: str, api_key: str = None, timeout: int = 90) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "input": prompt,
        "instructions": ""  # 你原本 API 需要的欄位，可自行調整
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["text"]


def try_generate(api_url: str, n: int, api_key: str, retries: int = 3, delay: int = 5) -> List[Dict[str, Any]]:
    prompt = build_prompt(n)
    for attempt in range(1, retries + 1):
        try:
            raw = call_llm(api_url, prompt, api_key)
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
        except requests.RequestException as e:
            print(f"⚠️  HTTP error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Failed to generate valid JSON after retries")


def main():
    ap = argparse.ArgumentParser(description="Generate fundraising events via ChatGPT API")
    ap.add_argument("--num_events", type=int, default=50, help="Number of events to generate")
    ap.add_argument("--out_file", default="events_list.json", help="Output JSON file")
    ap.add_argument("--api_url", default="http://57.129.18.204:51001/generate-text", help="LLM endpoint")
    ap.add_argument("--api_key", default=None, help="Optional API key for Authorization header")
    args = ap.parse_args()

    try:
        events = try_generate(args.api_url, args.num_events, args.api_key)
    except Exception as e:
        print(f"❌  Generation failed: {e}")
        sys.exit(1)

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"✅  Generated {len(events)} events → {args.out_file}")


if __name__ == "__main__":
    main()
