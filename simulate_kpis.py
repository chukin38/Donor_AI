#!/usr/bin/env python3
"""Simulate fundraising KPIs using a local language model."""

import argparse
import json
import csv
import tqdm
from transformers import pipeline
import variants  # assumes variants.py defines `variants` list

def simulate_kpi(variant, generator):
    """Generate simulated KPI metrics for one strategy variant."""
    prompt = (
        f"Estimate these KPI metrics for the fundraising strategy:\n"
        f"- Ask Amount: {variant['ask']}\n"
        f"- Format: {variant['format']}\n"
        f"- Tone: {variant['tone']}\n"
        f"- Channel: {variant['channel']}\n"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate KPIs with a local model")
    parser.add_argument("--model", default="gpt2", help="HF model name or path")
    args = parser.parse_args()

    generator = pipeline("text-generation", model=args.model)

    rows = []
    print("🔁 Running KPI simulation for each variant…")
    for v in tqdm.tqdm(variants.variants[:150], desc="Simulating"):
        kpi = simulate_kpi(v, generator)
        v.update(kpi)
        # Composite score
        revenue = kpi["avg_gift_hkd"] * kpi["conv_rate"]
        v["score"] = round(
            0.4 * revenue +
            0.25 * kpi["conv_rate"] +
            0.2 * kpi["retention_pct"] +
            0.15 * kpi["rsvp_pct"], 4
        )
        rows.append(v)

    if rows:
        keys = list(rows[0].keys())
        with open("simulation_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Saved simulation_results.csv ({len(rows)} rows)")
