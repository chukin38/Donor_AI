#!/usr/bin/env python3
"""Generate synthetic donor profiles using the OpenAI API."""

import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI


def generate_csv(model: str, prompt: str, num_rows: int, max_retries: int = 3) -> pd.DataFrame:
    """Call the OpenAI chat completion API and return a DataFrame."""
    client = OpenAI()
    system_msg = {"role": "system", "content": "You are a data-generation engine. Output ONLY CSV."}
    user_msg = {"role": "user", "content": prompt.replace("{NUM_ROWS}", str(num_rows))}

    for _ in range(max_retries):
        rsp = client.chat.completions.create(model=model, messages=[system_msg, user_msg], temperature=0.4)
        text = rsp.choices[0].message.content
        if "```csv" in text:
            csv_block = text.split("```csv")[1].split("```")[0].strip()
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
    ap = argparse.ArgumentParser(description="Generate synthetic donor dataset via OpenAI")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
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
