#!/usr/bin/env python3
"""Generate synthetic donor profiles using a local Gemma model."""

import argparse
from pathlib import Path

import pandas as pd
from transformers import pipeline

_GENERATOR = None


def generate_csv(model: str, prompt: str, num_rows: int, max_retries: int = 3) -> pd.DataFrame:
    """Call a local text-generation model and return a DataFrame."""
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
    assert df["Name"].is_unique, "Names must be unique"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic donor dataset via a local Gemma model")
    ap.add_argument(
        "--model",
        default="/Users/solomonchu/PycharmProjects/Project_Donor/gemma-3-4b-pt",
        help="HF model name or path",
    )
    ap.add_argument("--num_rows", type=int, default=100, help="Number of donor rows")
    ap.add_argument("--out_file", default="synthetic_donors.csv", help="Output CSV file")
    args = ap.parse_args()

    prompt = Path("donor_schema_prompt.txt").read_text()
    df = generate_csv(args.model, prompt, args.num_rows)

    columns = [
        "Name", "Age", "Gender", "Location", "Household_Income", "Education_Level", "Occupation", "Industry_Sector",
        "Marital_Status", "Parental_Status", "Ethnicity", "Language_Preference", "Religion", "Political_Affiliation",
        "Lifetime_Donation_Amount", "Average_Gift", "First_Gift_Date", "Last_Gift_Date", "Donation_Frequency",
        "Preferred_Donation_Channel", "Payment_Method", "Recurring_Donor", "Employer_Matching_Eligible",
        "Cause_Interest", "Secondary_Cause_Interest", "Event_Attendance", "Volunteer_Hours", "Email_Open_Rate",
        "Social_Media_Engagement", "Communication_Pref", "Primary_Cause_Interest", "Hobbies_Interests", "Life_Stage",
        "Previous_Nonprofit_Affiliations", "Values_Alignment", "Estimated_Net_Worth", "Donor_LTV_Score",
        "Major_Gift_Likelihood", "Donation_History"
    ]
    validate(df, columns, args.num_rows)

    Path("output").mkdir(exist_ok=True)
    df.to_csv(Path("output") / args.out_file, index=False)
    print(f"✅ Generated {len(df)} donors → output/{args.out_file}")
