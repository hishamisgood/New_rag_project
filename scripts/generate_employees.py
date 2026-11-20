"""
Generate 2,000 synthetic employees using Faker → data/employees.xlsx
Columns: id, first_name, last_name, sex, salary, years_of_expertise, role,
         department, absent_days_jan..dec, sick_leave_days, maternity_leave_days,
         paternity_leave_days, paid_time_off_days, mental_health_leave_days,
         hire_date, work_mode, office_location, health_plan, performance_rating,
         password
"""

import os
import random
import string
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, EXCEL_PATH

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

NUM_EMPLOYEES = 2000

# ── Realistic tech-company department weights ────────────────────────────────
DEPARTMENTS = [
    "Engineering", "Sales", "Product", "Operations", "Customer Support",
    "Marketing", "Research & Development", "Finance", "Human Resources", "Legal",
]
DEPT_WEIGHTS = [0.28, 0.16, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05, 0.03, 0.03]

# ── Roles by department with their seniority tier ────────────────────────────
# Format: (role_name, tier)
# Tiers: Junior | Mid | Senior | Lead/Manager | Director/VP
ROLES_BY_DEPARTMENT = {
    "Engineering": [
        ("Junior Software Engineer",   "Junior"),
        ("Software Engineer",          "Mid"),
        ("Senior Software Engineer",   "Senior"),
        ("Staff Engineer",             "Senior"),
        ("Tech Lead",                  "Lead/Manager"),
        ("Engineering Manager",        "Lead/Manager"),
        ("DevOps Engineer",            "Mid"),
        ("QA Engineer",                "Mid"),
        ("Data Engineer",              "Mid"),
        ("Frontend Developer",         "Mid"),
        ("Backend Developer",          "Mid"),
        ("VP of Engineering",          "Director/VP"),
    ],
    "Human Resources": [
        ("HR Coordinator",             "Junior"),
        ("HR Specialist",              "Mid"),
        ("Recruiter",                  "Mid"),
        ("HR Business Partner",        "Senior"),
        ("Talent Acquisition Lead",    "Lead/Manager"),
        ("HR Manager",                 "Lead/Manager"),
        ("Compensation Analyst",       "Mid"),
        ("Training Coordinator",       "Junior"),
        ("HR Director",                "Director/VP"),
    ],
    "Marketing": [
        ("Marketing Specialist",       "Mid"),
        ("Content Writer",             "Junior"),
        ("SEO Specialist",             "Mid"),
        ("Social Media Manager",       "Mid"),
        ("Digital Marketing Analyst",  "Mid"),
        ("Brand Strategist",           "Senior"),
        ("Marketing Manager",          "Lead/Manager"),
        ("Marketing Director",         "Director/VP"),
    ],
    "Finance": [
        ("Financial Analyst",          "Mid"),
        ("Accountant",                 "Junior"),
        ("Senior Accountant",          "Senior"),
        ("Tax Specialist",             "Mid"),
        ("Auditor",                    "Mid"),
        ("Treasury Analyst",           "Mid"),
        ("Finance Manager",            "Lead/Manager"),
        ("Controller",                 "Director/VP"),
        ("CFO",                        "Director/VP"),
    ],
    "Sales": [
        ("Business Development Rep",   "Junior"),
        ("Inside Sales Rep",           "Junior"),
        ("Sales Representative",       "Mid"),
        ("Account Executive",          "Mid"),
        ("Sales Engineer",             "Senior"),
        ("Regional Sales Manager",     "Lead/Manager"),
        ("Sales Manager",              "Lead/Manager"),
        ("Sales Director",             "Director/VP"),
        ("VP of Sales",                "Director/VP"),
    ],
    "Operations": [
        ("Logistics Coordinator",      "Junior"),
        ("Operations Analyst",         "Mid"),
        ("Supply Chain Analyst",       "Mid"),
        ("Process Improvement Specialist", "Senior"),
        ("Project Manager",            "Senior"),
        ("Operations Manager",         "Lead/Manager"),
        ("Operations Director",        "Director/VP"),
    ],
    "Legal": [
        ("Legal Assistant",            "Junior"),
        ("Paralegal",                  "Junior"),
        ("Contract Specialist",        "Mid"),
        ("Compliance Officer",         "Senior"),
        ("IP Attorney",                "Senior"),
        ("Legal Counsel",              "Senior"),
        ("Senior Legal Counsel",       "Lead/Manager"),
        ("General Counsel",            "Director/VP"),
    ],
    "Product": [
        ("Product Analyst",            "Junior"),
        ("UX Researcher",              "Mid"),
        ("Product Designer",           "Mid"),
        ("Product Owner",              "Mid"),
        ("Product Manager",            "Senior"),
        ("Senior Product Manager",     "Lead/Manager"),
        ("VP of Product",              "Director/VP"),
    ],
    "Customer Support": [
        ("Help Desk Analyst",          "Junior"),
        ("Support Specialist",         "Junior"),
        ("Technical Support Engineer", "Mid"),
        ("Customer Success Manager",   "Senior"),
        ("Support Team Lead",          "Lead/Manager"),
        ("Support Manager",            "Lead/Manager"),
    ],
    "Research & Development": [
        ("Lab Technician",             "Junior"),
        ("Research Scientist",         "Mid"),
        ("ML Engineer",                "Mid"),
        ("Data Scientist",             "Senior"),
        ("Senior Researcher",          "Senior"),
        ("R&D Manager",                "Lead/Manager"),
        ("Research Director",          "Director/VP"),
    ],
}

# ── Pyramid weights: bottom-heavy org ────────────────────────────────────────
# (Junior, Mid, Senior, Lead/Manager, Director/VP)
TIER_PYRAMID_WEIGHTS = {
    "Engineering":          [0.25, 0.40, 0.22, 0.10, 0.03],
    "Human Resources":      [0.20, 0.35, 0.25, 0.15, 0.05],
    "Marketing":            [0.20, 0.38, 0.22, 0.15, 0.05],
    "Finance":              [0.22, 0.38, 0.22, 0.13, 0.05],
    "Sales":                [0.25, 0.38, 0.17, 0.15, 0.05],
    "Operations":           [0.22, 0.38, 0.22, 0.13, 0.05],
    "Legal":                [0.20, 0.35, 0.28, 0.12, 0.05],
    "Product":              [0.20, 0.35, 0.25, 0.15, 0.05],
    "Customer Support":     [0.35, 0.35, 0.18, 0.12, 0.00],
    "Research & Development": [0.20, 0.38, 0.25, 0.12, 0.05],
}

# ── Years-of-expertise range per tier ────────────────────────────────────────
YOE_RANGES = {
    "Junior":       (0, 2),
    "Mid":          (2, 5),
    "Senior":       (5, 10),
    "Lead/Manager": (8, 15),   # managers must have >=2 yrs (and typically 8+)
    "Director/VP":  (12, 25),
}

# ── Salary ranges by tier (base, before dept multiplier) ────────────────────
SALARY_RANGES = {
    "Junior":       (38_000,  65_000),
    "Mid":          (60_000,  95_000),
    "Senior":       (90_000, 130_000),
    "Lead/Manager": (110_000, 165_000),
    "Director/VP":  (150_000, 230_000),
}

# ── Entry/support role salary cap $150k ─────────────────────────────────────
ENTRY_SUPPORT_ROLES = {
    "Legal Assistant", "Paralegal", "Lab Technician",
    "Help Desk Analyst", "Support Specialist",
    "HR Coordinator", "Training Coordinator",
    "Logistics Coordinator", "Content Writer",
}
ENTRY_SALARY_CAP = 150_000

# ── Department salary multipliers ────────────────────────────────────────────
DEPT_MULTIPLIERS = {
    "Engineering":          1.10,
    "Research & Development": 1.08,
    "Product":              1.06,
    "Finance":              1.04,
    "Sales":                1.02,
    "Legal":                1.05,
    "Marketing":            0.98,
    "Operations":           0.97,
    "Human Resources":      0.96,
    "Customer Support":     0.94,
}

TIERS = ["Junior", "Mid", "Senior", "Lead/Manager", "Director/VP"]

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Seasonality weights for absence distribution (higher in summer and winter)
# Indices: Jan=0 ... Dec=11
ABSENCE_SEASON_WEIGHTS = [
    0.11, 0.07, 0.07, 0.07, 0.07, 0.08,   # Jan-Jun
    0.11, 0.10, 0.07, 0.07, 0.07, 0.11,   # Jul-Dec
]

# Reference date for hire_date derivation
REFERENCE_DATE = date(2026, 10, 1)


def _pick_tier(dept: str) -> str:
    return random.choices(TIERS, weights=TIER_PYRAMID_WEIGHTS[dept], k=1)[0]


def _pick_role(dept: str, tier: str):
    """Pick a role matching the given tier; fallback to any if none exist."""
    candidates = [r for r, t in ROLES_BY_DEPARTMENT[dept] if t == tier]
    if not candidates:
        candidates = [r for r, _ in ROLES_BY_DEPARTMENT[dept]]
    return random.choice(candidates)


def _yoe_for_tier(tier: str) -> int:
    lo, hi = YOE_RANGES[tier]
    return random.randint(lo, hi)


def _salary_for_tier_and_dept(tier: str, dept: str, role: str) -> float:
    lo, hi = SALARY_RANGES[tier]
    mult = DEPT_MULTIPLIERS.get(dept, 1.0)
    sal = random.uniform(lo * mult, hi * mult)
    # Cap entry/support roles
    if role in ENTRY_SUPPORT_ROLES:
        sal = min(sal, ENTRY_SALARY_CAP)
    return round(sal, 2)


def _pto_for_tenure(yoe: int) -> int:
    """PTO days per leave_policy_report.pdf: 0-1y=15, 2-4y=18, 5-9y=21, 10+y=25."""
    if yoe < 2:
        return 15
    if yoe < 5:
        return 18
    if yoe < 10:
        return 21
    return 25


def _hire_date_for_yoe(yoe: int) -> str:
    """Compute hire date consistent with years_of_expertise relative to 2026-10-01."""
    # Scatter within the year band
    extra_days = random.randint(0, 364)
    days_ago = yoe * 365 + extra_days
    hd = REFERENCE_DATE - timedelta(days=days_ago)
    return hd.strftime("%Y-%m-%d")


def _distribute_with_seasonality(total: int, max_per_month: int = 22) -> list[int]:
    """
    Distribute *total* leave days across 12 months using seasonal weights,
    capping any single month at max_per_month.
    """
    if total == 0:
        return [0] * 12
    monthly = [0] * 12
    remaining = total
    for _ in range(remaining):
        # re-weight by remaining capacity
        caps = [max(0, max_per_month - monthly[i]) for i in range(12)]
        total_cap = sum(caps)
        if total_cap == 0:
            break
        # combine seasonal weight with remaining capacity
        weights = [ABSENCE_SEASON_WEIGHTS[i] * caps[i] for i in range(12)]
        w_sum = sum(weights)
        if w_sum == 0:
            break
        probs = [w / w_sum for w in weights]
        chosen = random.choices(range(12), weights=probs, k=1)[0]
        monthly[chosen] += 1
    return monthly


def _place_parental_leave_block(monthly: list[int], block_days: int, max_per_month: int = 22) -> list[int]:
    """
    Place parental leave in consecutive monthly blocks.
    Picks a random start month and fills forward, respecting cap.
    """
    if block_days == 0:
        return monthly
    start = random.randint(0, 11)
    remaining = block_days
    idx = start
    for _ in range(12):
        if remaining <= 0:
            break
        available = max(0, max_per_month - monthly[idx % 12])
        take = min(remaining, available)
        monthly[idx % 12] += take
        remaining -= take
        idx += 1
    return monthly


def _unique_password(used: set, length: int = 12) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%&"
    while True:
        pwd = "".join(random.choices(chars, k=length))
        if pwd not in used:
            used.add(pwd)
            return pwd


def generate_employees() -> pd.DataFrame:
    """Create 2 000 employee records and save to Excel."""
    records: list[dict] = []
    used_passwords: set[str] = set()

    for i in range(1, NUM_EMPLOYEES + 1):
        emp_id = f"EMP{i:05d}"
        sex = random.choice(["Male", "Female"])
        first = fake.first_name_male() if sex == "Male" else fake.first_name_female()
        last = fake.last_name()

        # --- Department (weighted) + Role (tier-mapped) ---
        dept = random.choices(DEPARTMENTS, weights=DEPT_WEIGHTS, k=1)[0]
        tier = _pick_tier(dept)
        role = _pick_role(dept, tier)
        yoe  = _yoe_for_tier(tier)
        salary = _salary_for_tier_and_dept(tier, dept, role)

        # --- PTO per policy ---
        pto = _pto_for_tenure(yoe)

        # --- Other leave totals ---
        sick   = random.randint(0, 15)
        mental = random.choices([0, 1, 2, 3, 4, 5],
                                weights=[50, 20, 15, 8, 5, 2])[0]
        maternity = (random.choices([0, 30, 60, 90],
                                    weights=[70, 10, 10, 10])[0]
                     if sex == "Female" else 0)
        paternity = (random.choices([0, 5, 10, 15],
                                    weights=[60, 20, 10, 10])[0]
                     if sex == "Male" else 0)

        # --- Distribute non-parental days with seasonality ---
        non_parental = sick + pto + mental
        monthly = _distribute_with_seasonality(non_parental, max_per_month=22)

        # --- Parental leave in consecutive blocks ---
        if maternity > 0:
            monthly = _place_parental_leave_block(monthly, maternity, max_per_month=22)
        if paternity > 0:
            monthly = _place_parental_leave_block(monthly, paternity, max_per_month=22)

        # Enforce monthly cap (safety clamp)
        monthly = [min(d, 22) for d in monthly]

        # --- New RAG cross-reference fields ---
        hire_date = _hire_date_for_yoe(yoe)

        # work_mode: Remote-Exempt for fully remote roles/locations, Hybrid default
        work_mode = random.choices(
            ["Hybrid", "Remote-Exempt", "In-Office"],
            weights=[0.55, 0.25, 0.20],
            k=1,
        )[0]

        # office_location: Remote for Remote-Exempt
        if work_mode == "Remote-Exempt":
            office_location = "Remote"
        else:
            office_location = random.choices(
                ["San Francisco HQ", "New York", "Austin"],
                weights=[0.55, 0.25, 0.20],
                k=1,
            )[0]

        # health_plan: higher tiers skew Premium/Plus
        if tier in ("Director/VP", "Lead/Manager"):
            hp_weights = [0.20, 0.40, 0.40]
        elif tier == "Senior":
            hp_weights = [0.35, 0.40, 0.25]
        else:
            hp_weights = [0.55, 0.30, 0.15]
        health_plan = random.choices(
            ["Standard Plan", "Premium Plan", "Premium Plus"],
            weights=hp_weights,
            k=1,
        )[0]

        # performance_rating: 1-5, weighted around 3-4
        performance_rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.03, 0.08, 0.30, 0.42, 0.17],
            k=1,
        )[0]

        password = _unique_password(used_passwords)

        row = {
            "id":                    emp_id,
            "first_name":            first,
            "last_name":             last,
            "sex":                   sex,
            "department":            dept,
            "role":                  role,
            "years_of_expertise":    yoe,
            "salary":                salary,
        }
        for j, m in enumerate(MONTHS):
            row[f"absent_days_{m}"] = monthly[j]

        row["sick_leave_days"]          = sick
        row["maternity_leave_days"]     = maternity
        row["paternity_leave_days"]     = paternity
        row["paid_time_off_days"]       = pto
        row["mental_health_leave_days"] = mental

        # Cross-reference fields for multi-hop RAG
        row["hire_date"]           = hire_date
        row["work_mode"]           = work_mode
        row["office_location"]     = office_location
        row["health_plan"]         = health_plan
        row["performance_rating"]  = performance_rating

        row["password"] = password

        records.append(row)

    df = pd.DataFrame(records)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")

    print(f"✅ Generated {len(df)} employees → {EXCEL_PATH}")
    print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\n   Department distribution:\n{df['department'].value_counts().to_string()}")
    print(f"\n   Tier/seniority check (via years_of_expertise):")
    print(f"     yoe < 2:   {(df['years_of_expertise'] < 2).sum()}")
    print(f"     yoe 2-4:   {((df['years_of_expertise'] >= 2) & (df['years_of_expertise'] < 5)).sum()}")
    print(f"     yoe 5-9:   {((df['years_of_expertise'] >= 5) & (df['years_of_expertise'] < 10)).sum()}")
    print(f"     yoe 10-14: {((df['years_of_expertise'] >= 10) & (df['years_of_expertise'] < 15)).sum()}")
    print(f"     yoe 15+:   {(df['years_of_expertise'] >= 15).sum()}")
    print(f"\n   PTO range: {df['paid_time_off_days'].min()}–{df['paid_time_off_days'].max()}")
    print(f"   Max monthly absence: {df[[f'absent_days_{m}' for m in MONTHS]].max().max()}")
    print(f"\n   Sample (first 3 rows):\n{df.head(3).to_string()}")
    return df


if __name__ == "__main__":
    generate_employees()
