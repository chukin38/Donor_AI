#!/usr/bin/env python3
"""Generate fundraising event data using a local Gemma model."""

import argparse
from pathlib import Path

import pandas as pd
from transformers import pipeline

_GENERATOR = None


def generate_csv(model: str, prompt: str, num_rows: int, max_retries: int = 3) -> pd.DataFrame:
    """Call a local text-generation model and parse the returned CSV."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = pipeline("text-generation", model=model)

    formatted = prompt.replace("{NUM_ROWS}", str(num_rows))
    for _ in range(max_retries):
        result = _GENERATOR(formatted, max_new_tokens=2048, do_sample=False)
        text = result[0]["generated_text"]
        if text.startswith(formatted):
            text = text[len(formatted):]
        if "```csv" in text:
            csv_block = text.split("```csv")[1].split("```", 1)[0].strip()
            try:
                return pd.read_csv(pd.compat.StringIO(csv_block))
            except Exception as e:
                print("Parse error:", e)
    raise ValueError("Failed to obtain valid CSV.")


def validate(df: pd.DataFrame, columns: list, num_rows: int):
    assert list(df.columns) == columns, "Header mismatch"
    assert len(df) == num_rows, "Row count mismatch"
    assert df.isnull().sum().sum() == 0, "Null values detected"
    assert df["Event_Name"].is_unique, "Event names must be unique"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate fundraising events via a local Gemma model")
    ap.add_argument(
        "--model",
        default="/Users/solomonchu/PycharmProjects/Project_Donor/gemma-3-4b-pt",
        help="HF model name or path",
    )
    ap.add_argument("--num_rows", type=int, default=20, help="Number of events")
    ap.add_argument("--out_file", default="synthetic_events.csv", help="Output CSV file")
    args = ap.parse_args()

    prompt = Path("event_schema_prompt.txt").read_text()
    df = generate_csv(args.model, prompt, args.num_rows)

    columns = [
        "Event_Name", "Event_Type", "Cause_Focus", "Target_Audience", "Location", "Goal_Amount", "Ticket_Price", "Event_Date",
        "Description", "Organizer", "Event_Duration", "Expected_Attendance", "Sponsorship_Tiers", "VIP_Package_Price", "Dress_Code",
        "Language", "Catering_Type", "Entertainment", "Networking_Opportunities", "Media_Coverage", "Registration_Deadline",
        "Early_Bird_Discount", "Group_Discount", "Corporate_Sponsorship_Available", "Volunteer_Opportunities",
        "Accessibility_Features", "Parking_Available", "Public_Transport_Access", "Weather_Contingency", "Follow_Up_Events",
        "Impact_Metrics", "Previous_Year_Attendance", "Previous_Year_Funds_Raised", "Celebrity_Guests", "Keynote_Speakers",
        "Workshop_Sessions", "Silent_Auction", "Live_Auction", "Raffle_Prizes", "Photo_Opportunities", "Social_Media_Hashtag",
        "Live_Streaming", "Recording_Available", "Tax_Deductible", "Employer_Matching_Eligible", "Payment_Methods", "Refund_Policy",
        "Age_Restrictions", "Dietary_Accommodations", "Cultural_Considerations", "Sustainability_Initiatives"
    ]
    validate(df, columns, args.num_rows)

    Path("output").mkdir(exist_ok=True)
    df.to_csv(Path("output") / args.out_file, index=False)
    print(f"✅ Generated {len(df)} events → output/{args.out_file}")
