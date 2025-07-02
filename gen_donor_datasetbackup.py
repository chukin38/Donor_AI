#!/usr/bin/env python
# generate_donors.py
import random, math, datetime
from faker import Faker
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check

fake = Faker(["en_US"])
N = 1_000
today = datetime.date(2025, 6, 15)

# ---- 1. 產生隨機資料 ----
def random_age():
    # 假設年齡呈右偏，平均 52 歲
    return int(np.clip(np.random.normal(52, 15), 20, 85))

def income_by_age(age):
    base = np.random.lognormal(mean=11, sigma=0.5)      # 約 60k‑200k
    bump = 1.2 if age > 55 else 0.9
    return int(base * bump)

def draw_beta(a, b):
    return np.random.beta(a, b)

donors = []
for _ in range(N):
    age = random_age()
    income = income_by_age(age)
    lifetime = np.random.gamma(shape=2, scale=income/20)
    avg_gift = max(10, np.random.normal(lifetime/20, 50))
    last_date = fake.date_between_dates(
        date_start=datetime.date(2022, 1, 1), date_end=today
    )
    donors.append(
        dict(
            full_name=fake.name(),
            age=age,
            gender=random.choices(
                ["Male", "Female", "Non‑binary", "Prefer not to say"],
                weights=[0.4, 0.4, 0.05, 0.15],
            )[0],
            state=fake.state_abbr(),
            zip_code=fake.postcode(),
            education_level=random.choices(
                ["High School", "Bachelor", "Master", "PhD"],
                weights=[0.25, 0.45, 0.25, 0.05],
            )[0],
            religion=random.choice(
                ["Christian", "Catholic", "Jewish", "Muslim", "Buddhist", "None"]
            ),
            household_income=income,
            lifetime_donation_usd=round(lifetime, 2),
            average_gift_usd=round(avg_gift, 2),
            donation_frequency=random.choice(
                ["One‑off", "Annual", "Quarterly", "Monthly"]
            ),
            last_gift_date=last_date,
            preferred_channel=random.choice(["Email", "Mail", "Phone", "SMS", "Social"]),
            recurring_donor=False,  # 先填 False，下段程式再修正
            employer_matching=np.random.rand() < 0.15,
            events_attended=np.random.poisson(2),
            volunteer_hours=round(np.random.gamma(2, 5), 1),
            email_open_rate=round(draw_beta(2, 5), 2),
            communication_preference=random.choice(["Email", "SMS", "Phone", "Any"]),
            primary_cause=random.choice(
                ["Education", "Health", "Environment", "Arts", "Faith", "Poverty"]
            ),
            donor_ltv_pred=0.0,  # 佔位
            major_gift_score=0,  # 佔位
        )
    )

df = pd.DataFrame(donors)

# ---- 2. 衍生欄位修正 ----
df["recurring_donor"] = np.where(df["donation_frequency"] == "Monthly", True, False)
df["donor_ltv_pred"] = (df["lifetime_donation_usd"] * np.random.uniform(1.2, 3.0)).round(2)

def calc_major_gift(row):
    inc = row.household_income
    if inc > 250_000:
        return np.random.randint(80, 100)
    elif inc > 150_000:
        return np.random.randint(60, 80)
    elif inc > 90_000:
        return np.random.randint(40, 60)
    else:
        return np.random.randint(0, 40)
df["major_gift_score"] = df.apply(calc_major_gift, axis=1)

# ---- 3. 基本驗證 (pandera) ----
schema = pa.DataFrameSchema(
    {
        "full_name": Column(str),
        "age": Column(int, Check.between(20, 85)),
        "household_income": Column(int, Check.greater_than_or_equal_to(10_000)),
        "email_open_rate": Column(float, Check.in_range(0, 1)),
    }
)
schema.validate(df, lazy=True)

# ---- 4. 輸出 ----
df.to_csv("donors_1k.csv", index=False)
print("✅ Generated donors_1k.csv with", len(df), "rows")
