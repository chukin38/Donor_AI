"""
gen_donor_dataset.py
生成 1k 筆符合上表規則的假 Donor 資料，輸出 donors_fake.csv
完全離線執行，僅依賴免費開源套件：
    pip install pandas numpy faker
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
from pathlib import Path

fake = Faker(["en_US", "zh_CN"])   # 中英文姓名各 50 %

N = 1_000
now = datetime.today()

# ---------- 工具函式 ----------
def random_date_within_months(months_back: int):
    offset_days = random.randint(0, months_back * 30)
    return (now - timedelta(days=offset_days)).date()

def pick_weighted(options, probs):
    return np.random.choice(options, p=probs)

# ---------- 常量 ----------
STATES, state_probs = zip(*[
    ("CA", 0.12), ("TX", 0.09), ("FL", 0.06), ("NY", 0.06),
    ("PA", 0.04), ("IL", 0.04), ("OH", 0.04), ("GA", 0.03),
    ("NC", 0.03), ("MI", 0.03)
])
state_probs = np.array(state_probs)
state_probs = np.append(state_probs, 1 - state_probs.sum())
STATES  = STATES + ("Other",)

income_brackets  = ["<50k", "50-100k", "100-250k", ">250k"]
income_probs     = [0.25, 0.35, 0.30, 0.10]
income_to_avg_range = {
    "<50k": (30, 150),
    "50-100k": (80, 400),
    "100-250k": (200, 1000),
    ">250k": (500, 3000)
}

education_levels = ["HighSchool", "College", "Graduate"]
education_probs  = [0.20, 0.40, 0.40]

occupations = ["Tech", "Finance", "Healthcare", "Education", "Public", "Other"]
occ_probs   = [0.15, 0.12, 0.13, 0.10, 0.08, 0.42]

religions   = ["Christian", "Catholic", "Jewish", "Muslim", "None", "Other"]
relig_probs = [0.45, 0.20, 0.02, 0.01, 0.25, 0.07]

causes      = ["Education", "Healthcare", "Environment", "ChildDevelopment",
               "PovertyRelief", "AnimalWelfare", "Other"]
cause_probs = [0.20, 0.18, 0.15, 0.12, 0.15, 0.10, 0.10]

freqs       = ["Monthly", "Quarterly", "Yearly", "Sporadic"]
freq_probs  = [0.25, 0.10, 0.30, 0.35]

pref_channels = ["Email", "SMS", "DirectMail", "Phone"]
channel_probs = [0.60, 0.15, 0.15, 0.10]

# ---------- 生成 ----------
data = []
for _ in range(N):
    # 基本人口統計
    gender = pick_weighted(["Male", "Female", "Other"], [0.48, 0.50, 0.02])
    name   = fake.name_male() if gender == "Male" else fake.name_female()
    age    = int(np.clip(np.random.normal(45, 13), 18, 85))
    state  = pick_weighted(STATES, state_probs)

    # 社經
    income_br = pick_weighted(income_brackets, income_probs)
    education = pick_weighted(education_levels, education_probs)
    occupation= pick_weighted(occupations, occ_probs)
    religion  = pick_weighted(religions, relig_probs)

    # 捐贈偏好
    primary_cause = pick_weighted(causes, cause_probs)
    donation_freq = pick_weighted(freqs, freq_probs)

    avg_low, avg_high = income_to_avg_range[income_br]
    avg_gift = round(np.random.lognormal(
        mean=np.log((avg_low + avg_high)/2),
        sigma=0.4), 2)
    avg_gift = np.clip(avg_gift, avg_low, avg_high)

    # 終身捐贈額估算
    frequency_factor = {"Monthly": 12, "Quarterly": 4,
                        "Yearly": 1, "Sporadic": np.random.randint(1, 3)}
    years_donating = np.random.randint(1, 11)
    lifetime_donation = round(avg_gift *
                              frequency_factor[donation_freq] *
                              years_donating, 2)

    # 近期行為
    last_gift_date = random_date_within_months(36)
    recurring_bool = (donation_freq in ["Monthly", "Quarterly"] and
                      np.random.rand() < 0.70) or \
                     (donation_freq in ["Yearly", "Sporadic"] and
                      np.random.rand() < 0.15)

    pref_channel   = pick_weighted(pref_channels, channel_probs)
    comm_pref      = pref_channel  # demo: 同步處理，亦可隨機多選
    event_attend   = np.random.poisson(1.2)
    volunteer_h    = 0 if np.random.rand() < 0.80 else \
                     np.random.randint(1, 21) if np.random.rand() < 0.75 else \
                     np.random.randint(21, 101)
    email_engage   = round(np.random.beta(2, 5), 3)
    employer_match = bool(np.random.rand() < 0.20) if occupation != "Other" else False

    data.append(dict(
        name=name,
        age=age,
        gender=gender,
        state=state,
        income_bracket=income_br,
        education_level=education,
        occupation_sector=occupation,
        religion=religion,
        primary_cause=primary_cause,
        donation_frequency=donation_freq,
        avg_gift_usd=avg_gift,
        lifetime_donation_usd=lifetime_donation,
        last_gift_date=last_gift_date,
        recurring_donor=recurring_bool,
        preferred_channel=pref_channel,
        event_attendance_cnt=int(event_attend),
        volunteer_hours_yr=int(volunteer_h),
        email_engagement=email_engage,
        communication_pref=comm_pref,
        employer_matching=employer_match
    ))

df = pd.DataFrame(data)
Path(".venv/output").mkdir(exist_ok=True)
df.to_csv("output/donors_fake.csv", index=False)
print("✅ 生成完成：output/donors_fake.csv")
