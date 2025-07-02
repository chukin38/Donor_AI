#!/usr/bin/env python3
import argparse
import json
import csv
from typing import List, Dict
from pathlib import Path

import requests
import pandas as pd
import tqdm
import variants  # assumes variants.py defines `variants` list
from kpi_utils import adjust_with_baseline

# Configuration
GENERATIVE_SERVER_URL = "http://57.129.18.204:51001"

def simulate_kpi(variant):
    """Call LLM endpoint to predict KPIs for one strategy variant."""
    payload = {
        "input": (
            f"Estimate these KPI metrics for the fundraising strategy:\n"
            f"- Ask Amount: {variant['ask']}\n"
            f"- Format: {variant['format']}\n"
            f"- Tone: {variant['tone']}\n"
            f"- Channel: {variant['channel']}\n"
            "Return ONLY valid JSON with keys: rsvp_pct, conv_rate, avg_gift_hkd, retention_pct."
        ),
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

def run_simulation(
    variants_list: List[Dict],
    event: Dict | None,
    donors: pd.DataFrame | None,
    out_file: str,
):
    rows = []
    print("🔁 Running KPI simulation for each variant…")
    for v in tqdm.tqdm(variants_list[:150], desc="Simulating"):
        kpi = simulate_kpi(v)
        if event is not None and donors is not None:
            kpi = adjust_with_baseline(event, donors, kpi)
        v.update(kpi)
        # Composite score
        revenue = kpi["avg_gift_hkd"] * kpi["conv_rate"]
        v["score"] = round(
            0.4 * revenue
            + 0.25 * kpi["conv_rate"]
            + 0.2 * kpi["retention_pct"]
            + 0.15 * kpi["rsvp_pct"],
            4,
        )
        rows.append(v)

    if rows:
        keys = list(rows[0].keys())
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Saved {out_file} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate fundraising KPIs via LLM")
    ap.add_argument(
        "--proposals",
        default="variants.json",
        help="JSON file containing strategy variants",
    )
    ap.add_argument(
        "--output",
        default="simulation_results.csv",
        help="CSV file to store results",
    )
    ap.add_argument("--event_json", help="Event JSON (single event or list)")
    ap.add_argument("--donor_csv", help="CSV of selected donors")
    args = ap.parse_args()

    variants_list = variants.variants
    if args.proposals and args.proposals != "variants.json":
        with open(args.proposals, "r", encoding="utf-8") as f:
            variants_list = json.load(f)

    event = None
    donors = None
    if args.event_json and Path(args.event_json).exists():
        with open(args.event_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            event = data[0] if isinstance(data, list) else data
    if args.donor_csv and Path(args.donor_csv).exists():
        donors = pd.read_csv(args.donor_csv)

    run_simulation(variants_list, event, donors, args.output)
