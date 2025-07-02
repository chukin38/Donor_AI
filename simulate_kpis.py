#!/usr/bin/env python3
"""Simulate fundraising KPIs using a local language model."""

import argparse
import json
import csv
import random
import tqdm
from transformers import pipeline
import variants  # assumes variants.py defines `variants` list

def event_brief(e: dict) -> str:
    """Return a short description for logging/prompts."""
    return (
        f"{e.get('title', '')} | Cause: {e.get('cause', '')} | "
        f"Goal: HK${e.get('goal_amount', e.get('goal', 'N/A'))} | "
        f"Date: {e.get('date_start', e.get('date', ''))}"
    )


def simulate_kpi(variant: dict, generator, event: dict, donor_scale: int) -> dict:
    """Generate KPI estimates for one strategy variant."""
    prompt = (
        f"Estimate these KPI metrics for the fundraising strategy:\n"
        f"- Ask Amount: {variant['ask']}\n"
        f"- Format: {variant['format']}\n"
        f"- Tone: {variant['tone']}\n"
        f"- Channel: {variant['channel']}\n"
        f"- Event: {event_brief(event)}\n"
        f"- Donors Contacted: {donor_scale}\n"
        "Return ONLY valid JSON with keys: rsvp_pct, conv_rate, avg_gift_hkd, retention_pct."
    )
    try:
        result = generator(prompt, max_new_tokens=128, do_sample=False)
        text = result[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):]
        return json.loads(text.strip())
    except Exception as e:
        print(f"❌ Error simulating {variant}: {e}")
        return {"rsvp_pct": 0, "conv_rate": 0, "avg_gift_hkd": 0, "retention_pct": 0}


def load_event(path: str, event_id: str) -> dict:
    """Return the event record matching ``event_id`` from ``path``."""
    events = json.load(open(path, "r", encoding="utf-8"))
    for e in events:
        if str(e.get("event_id")) == str(event_id):
            return e
    raise ValueError(f"Event ID {event_id} not found in {path}")


def load_donors(path: str, ids=None, count=None) -> list:
    """Return donor rows from ``path`` for the given indices or random count."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if ids:
        rows = [rows[i] for i in ids if 0 <= i < len(rows)]
    elif count:
        rows = random.sample(rows, min(count, len(rows)))
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate KPIs for strategy variants"
    )
    parser.add_argument("--model", default="gpt2", help="HF model name or path")
    parser.add_argument("--events_json", required=True, help="Path to events JSON list")
    parser.add_argument("--event_id", required=True, help="Event ID to simulate")
    parser.add_argument("--donors_csv", required=True, help="Path to donors CSV")
    parser.add_argument("--donor_ids", help="Comma-separated donor row indices")
    parser.add_argument("--donor_count", type=int, help="Randomly sample N donors")
    args = parser.parse_args()

    generator = pipeline("text-generation", model=args.model)

    event = load_event(args.events_json, args.event_id)
    donor_ids = [int(x) for x in args.donor_ids.split(",") if x.strip()] if args.donor_ids else None
    donors = load_donors(args.donors_csv, donor_ids, args.donor_count)
    donor_scale = len(donors)

    rows = []
    print("🔁 Running KPI simulation for each variant…")
    for v in tqdm.tqdm(variants.variants[:150], desc="Simulating"):
        kpi = simulate_kpi(v, generator, event, donor_scale)
        v.update(kpi)
        revenue = kpi["avg_gift_hkd"] * kpi["conv_rate"]
        v["score"] = round(
            0.4 * revenue +
            0.25 * kpi["conv_rate"] +
            0.2 * kpi["retention_pct"] +
            0.15 * kpi["rsvp_pct"],
            4,
        )
        rows.append(v)

    if rows:
        keys = list(rows[0].keys())
        with open("simulation_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Saved simulation_results.csv ({len(rows)} rows)")
