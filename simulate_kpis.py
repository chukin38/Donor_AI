#!/usr/bin/env python3
"""Simulate fundraising KPIs using a local language model."""

import argparse
import json
import csv
import tqdm
from transformers import pipeline
import variants  # assumes variants.py defines `variants` list
import pandas as pd
import os

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

        def generate_summary(result_csv: str, donors_csv: str = "output/donors_fake.csv",
                             summary_file: str = "simulation_summary.md"):
            """Create a markdown summary comparing best strategy to baseline."""
            if not os.path.exists(result_csv):
                print(f"❌ {result_csv} not found")
                return
            df = pd.read_csv(result_csv)
            if df.empty:
                print("❌ No rows to summarize")
                return

            baseline = df.iloc[0]
            best = df.loc[df["score"].idxmax()]

            n_donors = "N/A"
            if os.path.exists(donors_csv):
                try:
                    n_donors = len(pd.read_csv(donors_csv))
                except Exception:
                    pass

            base_rev = baseline["avg_gift_hkd"] * baseline["conv_rate"]
            best_rev = best["avg_gift_hkd"] * best["conv_rate"]
            if isinstance(n_donors, int):
                base_rev *= n_donors
                best_rev *= n_donors
            diff_pct = ((best_rev - base_rev) / base_rev * 100) if base_rev else 0

            lines = ["# KPI Simulation Summary\n",
                     f"**Selected Strategy:** Ask {best['ask']} | {best['format']} | {best['tone']} | {best['channel']}\n",
                     f"**Number of donors considered:** {n_donors}\n",
                     "## KPI Metrics\n",
                     f"- RSVP Rate: {best['rsvp_pct']}%\n",
                     f"- Conversion Rate: {best['conv_rate']}%\n",
                     f"- Avg Gift: HK${best['avg_gift_hkd']}\n",
                     f"- Retention: {best['retention_pct']}%\n",
                     f"- Predicted Revenue: HK${best_rev:.2f}\n\n",
                     "## Comparison with Baseline\n",
                     f"Baseline (Ask {baseline['ask']} | {baseline['format']} | {baseline['tone']} | {baseline['channel']}): HK${base_rev:.2f}\n",
                     f"Revenue uplift vs baseline: {diff_pct:.2f}%\n"]
            with open(summary_file, "w", encoding="utf-8") as sf:
                sf.writelines(lines)
            print(f"✅ Summary report → {summary_file}")

        generate_summary("simulation_results.csv")
