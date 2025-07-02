#!/usr/bin/env python3
# generate_events.py
"""Generate sample events using a local Gemma model."""

import json
import sys
from transformers import pipeline

# Configuration
MODEL_NAME = "gemma-4b"
_GENERATOR = pipeline("text-generation", model=MODEL_NAME)

def generate_event_list(num_events: int = 50):
    """
    呼叫 LLM 產生指定數量的捐款活動清單（JSON 字串）。
    """

    # ---------- ① 只改這裡 ----------
    prompt = (
        f"Generate a list of {num_events} charity fundraising events. Return ONLY valid JSON - an array of objects - no markdown, no code fences.\n"
        "Each event object MUST contain the following keys exactly: event_id, title, description, format, modality_ratio (object with in_person, online), venue (object with name, city, lat, lon), date_start, date_end, cause, sub_cause, keywords (array), target_segments (array), religious_affinity, age_range (array of two ints), language, goal_amount, donation_tiers (array of {label, min}), volunteer_slots, comm_channels (array), send_schedule (object with save_the_date, reminder), prev_years (array of {year, attendees, total_raised}), sponsor_names (array), matching_ratio, tax_deductible. Values should resemble real Hong Kong / Macau charity events (e.g., Jockey Club, World Vision). Dates in 2025, goal_amount in HKD, and use ISO-8601 for all timestamps."
    )
    # ---------------------------------

    try:
        result = _GENERATOR(prompt, max_new_tokens=2048, do_sample=False)
        events_json_str = result[0]["generated_text"]
        if events_json_str.startswith(prompt):
            events_json_str = events_json_str[len(prompt):]
        events_json_str = events_json_str.strip()
        # 驗證是否為合法 JSON（若失敗會丟 JSONDecodeError）
        json.loads(events_json_str)
        return events_json_str

    except (json.JSONDecodeError, Exception) as e:
        return f"Error generating events list: {e}"

# ---------- ② 只改這裡 ----------
if __name__ == "__main__":
    # Generate a report
    events_json = generate_event_list(50)

    # Save to file
    with open("events_list.json", "w", encoding="utf-8") as f:
        f.write(events_json)

    print("Events JSON generated and saved to 'events_list.json'")
    print(f"Characters: {len(events_json)}")
