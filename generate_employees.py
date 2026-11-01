"""
Generate 2,000 synthetic employees using Faker → data/employees.xlsx
Columns: id, first_name, last_name, sex, salary, years_of_expertise, role,
         department, absent_days_jan..dec, sick_leave_days, maternity_leave_days,
         paternity_leave_days, paid_time_off_days, mental_health_leave_days, password
"""

import os
import random
import string

import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

NUM_EMPLOYEES = 2000

DEPARTMENTS = [
    "Engineering", "Human Resources", "Marketing", "Finance", "Sales",
    "Operations", "Legal", "Product", "Customer Support",
    "Research & Development",
]

ROLES_BY_DEPARTMENT = {
    "Engineering": [
        "Software Engineer", "Senior Software Engineer", "Staff Engineer",
        "Engineering Manager", "DevOps Engineer", "QA Engineer",
        "Data Engineer", "Frontend Developer", "Backend Developer", "Tech Lead",
    ],
    "Human Resources": [
        "HR Specialist", "HR Manager", "Recruiter", "HR Coordinator",
        "Talent Acquisition Lead", "HR Business Partner",
        "Compensation Analyst", "Training Coordinator",
    ],
    "Marketing": [
        "Marketing Specialist", "Marketing Manager", "Content Writer",
        "SEO Specialist", "Social Media Manager", "Brand Strategist",
        "Digital Marketing Analyst", "Marketing Director",
    ],
    "Finance": [
        "Financial Analyst", "Accountant", "Senior Accountant",
        "Finance Manager", "Controller", "Tax Specialist",
        "Auditor", "Treasury Analyst",
    ],
    "Sales": [
        "Sales Representative", "Account Executive", "Sales Manager",
        "Business Development Rep", "Sales Director",
        "Regional Sales Manager", "Inside Sales Rep", "Sales Engineer",
    ],
    "Operations": [
        "Operations Analyst", "Operations Manager", "Project Manager",
        "Supply Chain Analyst", "Logistics Coordinator",
        "Operations Director", "Process Improvement Specialist",
    ],
    "Legal": [
        "Legal Counsel", "Paralegal", "Compliance Officer",
        "Legal Assistant", "Senior Legal Counsel",
        "Contract Specialist", "IP Attorney",
    ],
    "Product": [
        "Product Manager", "Senior Product Manager", "Product Analyst",
        "Product Designer", "UX Researcher", "Product Owner", "VP of Product",
    ],
    "Customer Support": [
        "Support Specialist", "Support Manager",
        "Technical Support Engineer", "Customer Success Manager",
        "Help Desk Analyst", "Support Team Lead",
    ],
    "Research & Development": [
        "Research Scientist", "Senior Researcher", "Lab Technician",
        "R&D Manager", "Data Scientist", "ML Engineer", "Research Director",
    ],
}

# Salary ranges by seniority tier
SALARY_RANGES = {
    "Junior":          (35_000, 55_000),
    "Mid":             (55_000, 85_000),
    "Senior":          (85_000, 120_000),
    "Lead/Manager":    (100_000, 160_000),
    "Director/VP":     (140_000, 220_000),
}

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def _seniority(years: int) -> str:
    if years < 2:
        return "Junior"
    if years < 5:
        return "Mid"
    if years < 10:
        return "Senior"
    if years < 15:
        return "Lead/Manager"
    return "Director/VP"


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
        dept = random.choice(DEPARTMENTS)
        role = random.choice(ROLES_BY_DEPARTMENT[dept])
        yoe = random.randint(0, 30)
        salary = round(random.uniform(*SALARY_RANGES[_seniority(yoe)]), 2)

        # --- leave totals for the year ---
        sick = random.randint(0, 15)
        pto = random.randint(5, 25)
        mental = random.choices([0, 1, 2, 3, 4, 5],
                                weights=[50, 20, 15, 8, 5, 2])[0]
        maternity = (random.choices([0, 30, 60, 90],
                                    weights=[70, 10, 10, 10])[0]
                     if sex == "Female" else 0)
        paternity = (random.choices([0, 5, 10, 15],
                                    weights=[60, 20, 10, 10])[0]
                     if sex == "Male" else 0)

        total_leave = sick + pto + mental + maternity + paternity

        # distribute across 12 months
        monthly = [0] * 12
        for _ in range(total_leave):
            monthly[random.randint(0, 11)] += 1

        password = _unique_password(used_passwords)

        row = {
            "id":                    emp_id,
            "first_name":            first,
            "last_name":             last,
            "sex":                   sex,
            "department":            dept,
            "role":                  role,
            "years_of_expertise":    yoe,
            "salary":               salary,
        }
        for j, m in enumerate(MONTHS):
            row[f"absent_days_{m}"] = monthly[j]

        row["sick_leave_days"]          = sick
        row["maternity_leave_days"]     = maternity
        row["paternity_leave_days"]     = paternity
        row["paid_time_off_days"]       = pto
        row["mental_health_leave_days"] = mental
        row["password"]                 = password

        records.append(row)

    df = pd.DataFrame(records)

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "employees.xlsx")
    df.to_excel(path, index=False, engine="openpyxl")

    print(f"✅ Generated {len(df)} employees → {path}")
    print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\n   Sample (first 3 rows):\n{df.head(3).to_string()}")
    return df


if __name__ == "__main__":
    generate_employees()
