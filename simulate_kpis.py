#!/usr/bin/env python3
import argparse
import requests
import json
import csv
import random
import tqdm
import variants  # assumes variants.py defines `variants` list

# Configuration
GENERATIVE_SERVER_URL = "http://57.129.18.204:51001"

def event_brief(e: dict) -> str:
    return (
        f"{e.get('title', '')} | Cause: {e.get('cause', '')} | "
        f"Goal: HK${e.get('goal_amount', e.get('goal', 'N/A'))} | "
        f"Date: {e.get('date_start', e.get('date', ''))}"
    )


def simulate_kpi(variant: dict, event: dict, donor_scale: int) -> dict:
    """Call LLM endpoint to predict KPIs for one strategy variant."""
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
    payload = {
        "input": prompt,
        "instructions": "Provide numeric values only. No extra text or markdown."
    }

    try:
        resp = requests.post(
            f"{GENERATIVE_SERVER_URL}/generate-text",
            json=payload,
            timeout=90
        )
        resp.raise_for_status()
        return json.loads(resp.json()["text"].strip())
    except requests.exceptions.RequestException as e:
        print(f"❌ Error simulating {variant}: {e}")
        return {"rsvp_pct": 0, "conv_rate": 0, "avg_gift_hkd": 0, "retention_pct": 0}


def load_event(path: str, event_id: str) -> dict:
    events = json.load(open(path, "r", encoding="utf-8"))
    for e in events:
        if str(e.get("event_id")) == str(event_id):
            return e
    raise ValueError(f"Event ID {event_id} not found in {path}")


def load_donors(path: str, ids=None, count=None) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if ids:
        rows = [rows[i] for i in ids if 0 <= i < len(rows)]
    elif count:
        rows = random.sample(rows, min(count, len(rows)))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate KPIs for strategy variants")
    ap.add_argument("--events_json", required=True, help="Path to events JSON list")
    ap.add_argument("--event_id", required=True, help="Event ID to simulate")
    ap.add_argument("--donors_csv", required=True, help="Path to donors CSV")
    ap.add_argument("--donor_ids", help="Comma-separated donor row indices")
    ap.add_argument("--donor_count", type=int, help="Randomly sample N donors")
    args = ap.parse_args()

    event = load_event(args.events_json, args.event_id)
    donor_ids = [int(x) for x in args.donor_ids.split(",") if x.strip()] if args.donor_ids else None
    donors = load_donors(args.donors_csv, donor_ids, args.donor_count)
    donor_scale = len(donors)

    rows = []
    print("🔁 Running KPI simulation for each variant…")
    for v in tqdm.tqdm(variants.variants[:150], desc="Simulating"):
        kpi = simulate_kpi(v, event, donor_scale)
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
