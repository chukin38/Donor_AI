#!/usr/bin/env python3
"""Estimate fundraising KPIs for a specific event using a local language model."""

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd
from transformers import pipeline

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_event(events_path: Path, event_id: str) -> dict:
    """Return the event dict with matching event_id."""
    events = json.load(events_path.open())
    for ev in events:
        if ev.get("event_id") == event_id:
            return ev
    raise ValueError(f"Event id {event_id} not found in {events_path}")


def baseline_estimate(event: dict, donor_count: int, scale: float) -> tuple:
    """Return (expected_donors, expected_revenue) based on past data."""
    if event.get("prev_years"):
        avg_attend = sum(y.get("attendees", 0) for y in event["prev_years"]) / len(event["prev_years"])
        avg_gift = sum(y.get("total_raised", 0) for y in event["prev_years"]) / (avg_attend * len(event["prev_years"]))
    else:
        avg_attend = event.get("goal_amount", 0) / 500
        avg_gift = 500
    est_donors = max(1, int(avg_attend * scale))
    est_revenue = int(est_donors * avg_gift)
    return est_donors, est_revenue


def llm_estimate(event: dict, donor_names: list, baseline: tuple, generator) -> dict:
    """Query the language model to refine KPI estimates."""
    sample_names = ", ".join(donor_names[:10])
    prompt = (
        f"Estimate KPIs for this charity event.\n"
        f"Title: {event.get('title')}\n"
        f"Cause: {event.get('cause')}\n"
        f"Goal Amount: HK${event.get('goal_amount')}\n"
        f"Baseline donors: {baseline[0]}, baseline revenue: HK${baseline[1]}.\n"
        f"Sample donors: {sample_names}\n"
        "Return ONLY valid JSON with keys: rsvp_pct, conv_rate, avg_gift_hkd, "
        "retention_pct, attendees, revenue."
    )

    try:
        result = generator(prompt, max_new_tokens=128, do_sample=False)
        text = result[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return json.loads(text.strip())
    except Exception as e:
        print(f"⚠️  LLM failed to produce JSON: {e}. Falling back to baseline.")
        return {
            "rsvp_pct": 60,
            "conv_rate": 40,
            "avg_gift_hkd": 500,
            "retention_pct": 50,
            "attendees": baseline[0],
            "revenue": baseline[1],
        }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate event KPIs with a local LLM")
    ap.add_argument("--events_json", required=True, help="Path to events JSON file")
    ap.add_argument("--event_id", required=True, help="ID of event to simulate")
    ap.add_argument("--donor_csv", default="output/donors_fake.csv", help="CSV of donors")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiplier for baseline attendees")
    ap.add_argument(
        "--model",
        default="/Users/solomonchu/PycharmProjects/Project_Donor/gemma-3-4b-pt",
        help="HuggingFace model name or path",
    )
    ap.add_argument("--out_csv", default="simulation_results.csv", help="Where to save KPI CSV")
    ap.add_argument("--report", default="event_report.txt", help="Where to save text report")
    args = ap.parse_args()

    event = load_event(Path(args.events_json), args.event_id)
    donors = pd.read_csv(args.donor_csv)
    donor_names = donors[donors.columns[0]].tolist()

    baseline = baseline_estimate(event, len(donors), args.scale)
    generator = pipeline("text-generation", model=args.model)
    kpi = llm_estimate(event, donor_names, baseline, generator)

    event_results = {**event, **kpi}
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_results.keys())
        writer.writeheader()
        writer.writerow(event_results)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"Event: {event['title']} ({event['event_id']})\n")
        f.write(f"Estimated attendees: {kpi['attendees']}\n")
        f.write(f"Expected revenue: HK${kpi['revenue']}\n")
        f.write(f"Conversion rate: {kpi['conv_rate']}%\n")
        f.write(f"RSVP rate: {kpi['rsvp_pct']}%\n")
        f.write(f"Retention: {kpi['retention_pct']}%\n")

    print(f"✅ Saved {args.out_csv} and {args.report}")
