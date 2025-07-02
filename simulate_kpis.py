#!/usr/bin/env python3
import requests
import json
import csv
import tqdm
import variants  # assumes variants.py defines `variants` list

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

if __name__ == "__main__":
    rows = []
    print("🔁 Running KPI simulation for each variant…")
    for v in tqdm.tqdm(variants.variants[:150], desc="Simulating"):
        kpi = simulate_kpi(v)
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
