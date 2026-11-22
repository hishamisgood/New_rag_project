"""SQLite schema setup and employee queries."""

import calendar
import random
import sqlite3
from contextlib import closing
from datetime import date

import pandas as pd

from .auth import hash_password
from .config import DB_PATH, EXCEL_PATH

INTEGER_COLUMNS = {
    "years_of_expertise",
    "performance_rating",
    "sick_leave_days",
    "maternity_leave_days",
    "paternity_leave_days",
    "paid_time_off_days",
    "mental_health_leave_days",
    "absent_days_jan", "absent_days_feb", "absent_days_mar",
    "absent_days_apr", "absent_days_may", "absent_days_jun",
    "absent_days_jul", "absent_days_aug", "absent_days_sep",
    "absent_days_oct", "absent_days_nov", "absent_days_dec",
}

TRANSACTIONAL_TABLES = (
    "expense_claims",
    "leave_requests",
    "payroll_history",
    "training_enrollments",
)

KNOWLEDGE_SEED = [
    ("it_support", "IT support",
     "For password, VPN, laptop, or access issues, open a ticket in the Service Desk portal. "
     "Critical account lockouts are handled by the IT Help Desk during business hours."),
    ("expenses", "Expense reimbursement",
     "Submit receipts and the completed expense report within 30 days of the purchase. "
     "Your manager must approve the report before Finance processes reimbursement."),
    ("payroll", "Payroll schedule",
     "NovaCorp pays employees monthly on the final business day. "
     "Questions about a payslip should be sent to Payroll through the HR service desk."),
    ("learning", "Learning and development",
     "Employees can request role-relevant training through their manager. "
     "Approved courses are recorded in the learning portal and reimbursed according to the approved budget."),
    ("holidays", "Company holidays",
     "NovaCorp observes 11 federal holidays per year plus 1 floating holiday employees may use "
     "at any time with manager approval. The holiday calendar is published each November for the "
     "following year and available on the HR Portal under Company Calendar."),
    ("badge_wifi", "Badge and Wi-Fi access",
     "New hire badge access is provisioned on Day 1 by IT Security. "
     "For lost badges or Wi-Fi connectivity issues, contact the IT Help Desk via the Service Desk portal "
     "or call the front-desk security office. Temporary access badges expire after 24 hours."),
    ("laptop_refresh", "Laptop refresh cycle",
     "NovaCorp refreshes employee laptops on a 3-year cycle. Requests for early refresh due to hardware "
     "failure must be submitted via the IT Service Desk with manager approval. "
     "Standard issue: MacBook Pro 14″ (Engineering/Product) or MacBook Air 13″ (all other roles)."),
    ("travel_perdiem", "Travel per-diem",
     "Employees on approved business travel receive a daily per-diem of $75 for meals and incidentals "
     "within the US, and $110 internationally. Hotel bookings must go through the corporate travel portal "
     "(Concur). Receipts for expenses above $25 must be submitted within 30 days."),
    ("referral_bonus", "Employee referral bonus",
     "NovaCorp pays a $3,000 referral bonus for each successful hire referred by a current employee. "
     "The bonus is paid in two instalments: $1,500 after the new hire's first month, "
     "and $1,500 after the new hire completes 6 months of employment. "
     "Referrals must be submitted through the HR Portal before the candidate applies."),
    ("security_incident", "Security incident reporting",
     "Any suspected data breach, phishing attempt, or security incident must be reported to the "
     "IT Security team within 24 hours by emailing security@novacorp.com or calling the security hotline. "
     "Do not attempt to investigate the incident yourself. Preserve all evidence (emails, screenshots) "
     "and avoid discussing the incident on unsecured channels."),
    ("hr_sla", "HR service-level agreement",
     "The HR service desk targets the following response SLAs: "
     "Tier 1 (general inquiries) – 1 business day; "
     "Tier 2 (payroll, benefits changes) – 3 business days; "
     "Tier 3 (investigations, accommodation requests) – 5 business days. "
     "Submit requests via the HR Portal under 'Submit a Request'."),
    ("remote_stipend", "Remote-work stipend",
     "Employees in approved Remote-Exempt roles receive a one-time home-office setup stipend of $1,000 "
     "and a monthly internet/utilities allowance of $75. "
     "The stipend is paid within 30 days of the remote designation being confirmed in Workday. "
     "Receipts are not required for the monthly allowance; the one-time stipend requires receipts."),
]


def _seed_expense_claims(
    conn: sqlite3.Connection,
    employee_id: str,
    rng: random.Random,
) -> None:
    categories = ["Travel", "Meals", "Equipment", "Training", "Books"]
    statuses = ["Pending", "Approved", "Paid", "Rejected"]
    descriptions = {
        "Travel": "Business trip transportation and lodging",
        "Meals": "Client meeting or business travel meal",
        "Equipment": "Role-related workstation equipment",
        "Training": "Approved professional development course",
        "Books": "Technical or business book purchase",
    }
    rows = []
    for claim_index in range(rng.randint(3, 6)):
        category = "Books" if claim_index == 0 else rng.choice(categories)
        amount = (
            round(rng.uniform(40, 280), 2)
            if category == "Books"
            else round(rng.uniform(35, 1450), 2)
        )
        month = rng.randint(1, 10)
        rows.append((
            employee_id,
            category,
            amount,
            f"2026-{month:02d}",
            rng.choice(statuses),
            descriptions[category],
        ))
    conn.executemany(
        "INSERT INTO expense_claims "
        "(employee_id, category, amount, submitted_date, status, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _seed_leave_requests(
    conn: sqlite3.Connection,
    employee_id: str,
    rng: random.Random,
) -> None:
    leave_types = ["PTO", "Sick", "Mental Health", "Volunteer"]
    statuses = ["Approved", "Pending", "Rejected"]
    rows = []
    for _ in range(rng.randint(2, 4)):
        month = rng.randint(1, 10)
        start_day = rng.randint(1, 24)
        days = rng.randint(1, 7)
        start = date(2026, month, start_day)
        end = date.fromordinal(start.toordinal() + days - 1)
        rows.append((
            employee_id, rng.choice(leave_types), start.isoformat(), end.isoformat(),
            days, rng.choice(statuses), "People Manager",
        ))
    conn.executemany(
        "INSERT INTO leave_requests "
        "(employee_id, leave_type, start_date, end_date, days, status, approver) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _seed_payroll_history(
    conn: sqlite3.Connection,
    employee_id: str,
    salary: float,
    health_plan: str,
    rating: int,
) -> None:
    bonus_rates = {1: 0.0, 2: 0.025, 3: 0.085, 4: 0.16, 5: 0.25}
    monthly_health_cost = {
        "Standard Plan": 90.0,
        "Premium Plan": 165.0,
        "Premium Plus": 240.0,
    }
    payroll_tax_rate = {1: 0.18, 2: 0.20, 3: 0.22, 4: 0.24, 5: 0.26}
    gross = round(salary / 12, 2)
    tax = round(gross * payroll_tax_rate.get(rating, 0.22), 2)
    benefits = monthly_health_cost.get(health_plan, 90.0)
    bonus = round(salary * bonus_rates.get(rating, 0.085), 2)
    rows = []
    for month in range(1, 11):
        last_day = calendar.monthrange(2026, month)[1]
        paid_date = date(2026, month, last_day)
        while paid_date.weekday() >= 5:
            paid_date = date.fromordinal(paid_date.toordinal() - 1)
        monthly_bonus = bonus if month == 7 else 0.0
        rows.append((
            employee_id, f"2026-{month:02d}", gross, tax, benefits, monthly_bonus,
            round(gross - tax - benefits + monthly_bonus, 2), paid_date.isoformat(),
        ))
    conn.executemany(
        "INSERT INTO payroll_history "
        "(employee_id, pay_month, gross_pay, tax_deducted, benefits_deducted, "
        "bonus, net_pay, paid_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _seed_training_enrollments(
    conn: sqlite3.Connection,
    employee_id: str,
    rng: random.Random,
) -> None:
    courses = [
        ("Cybersecurity Fundamentals", "Mandatory"),
        ("Project Management Professional", "Certification"),
        ("Industry Leadership Summit", "Conference"),
        ("Data Analytics with Python", "Online"),
        ("Cloud Architecture Essentials", "Certification"),
    ]
    statuses = ["Completed", "In Progress", "Approved"]
    enrollment_count = rng.randint(2, 4)
    raw_costs = [rng.uniform(100, 800) for _ in range(enrollment_count)]
    cost_scale = min(1.0, 2800 / sum(raw_costs))
    rows = []
    for index in range(enrollment_count):
        course_name, category = rng.choice(courses)
        month = rng.randint(1, 10)
        day = rng.randint(1, 28)
        rows.append((
            employee_id, course_name, category, round(raw_costs[index] * cost_scale, 2),
            rng.choice(statuses), date(2026, month, day).isoformat(),
        ))
    conn.executemany(
        "INSERT INTO training_enrollments "
        "(employee_id, course_name, category, cost, status, completion_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def ensure_transactional_tables() -> None:
    """Create and deterministically seed employee transaction tables."""
    rng = random.Random(42)
    table_sql = {
        "expense_claims": """
            CREATE TABLE IF NOT EXISTS expense_claims (
                id INTEGER PRIMARY KEY, employee_id TEXT NOT NULL, category TEXT,
                amount REAL, submitted_date TEXT, status TEXT, description TEXT
            )
        """,
        "leave_requests": """
            CREATE TABLE IF NOT EXISTS leave_requests (
                id INTEGER PRIMARY KEY, employee_id TEXT NOT NULL, leave_type TEXT,
                start_date TEXT, end_date TEXT, days INTEGER, status TEXT, approver TEXT
            )
        """,
        "payroll_history": """
            CREATE TABLE IF NOT EXISTS payroll_history (
                id INTEGER PRIMARY KEY, employee_id TEXT NOT NULL, pay_month TEXT,
                gross_pay REAL, tax_deducted REAL, benefits_deducted REAL, bonus REAL,
                net_pay REAL, paid_date TEXT
            )
        """,
        "training_enrollments": """
            CREATE TABLE IF NOT EXISTS training_enrollments (
                id INTEGER PRIMARY KEY, employee_id TEXT NOT NULL, course_name TEXT,
                category TEXT, cost REAL, status TEXT, completion_date TEXT
            )
        """,
    }
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        for table_name in TRANSACTIONAL_TABLES:
            conn.execute(table_sql[table_name])
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_employee_id "
                f"ON {table_name}(employee_id)"
            )

        employees = conn.execute(
            "SELECT id, salary, health_plan, performance_rating "
            "FROM employees ORDER BY id"
        ).fetchall()
        seeded_by_table = {
            table: {
                row[0] for row in conn.execute(
                    f"SELECT DISTINCT employee_id FROM {table}"
                ).fetchall()
            }
            for table in TRANSACTIONAL_TABLES
        }
        for employee_id, salary_value, health_plan, rating_value in employees:
            salary = float(salary_value or 60000)
            rating = int(rating_value or 3)
            if employee_id not in seeded_by_table["expense_claims"]:
                _seed_expense_claims(conn, employee_id, rng)
            if employee_id not in seeded_by_table["leave_requests"]:
                _seed_leave_requests(conn, employee_id, rng)
            if employee_id not in seeded_by_table["payroll_history"]:
                _seed_payroll_history(conn, employee_id, salary, health_plan, rating)
            if employee_id not in seeded_by_table["training_enrollments"]:
                _seed_training_enrollments(conn, employee_id, rng)
        conn.commit()


def init_db() -> None:
    """Create the employee table from the generated workbook."""
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS employees")
        cur.execute("DROP TABLE IF EXISTS rag_knowledge")

        column_definitions = []
        for column in df.columns:
            if column == "password":
                continue
            if column == "id":
                column_definitions.append("id TEXT PRIMARY KEY")
            elif column in INTEGER_COLUMNS:
                column_definitions.append(f'"{column}" INTEGER')
            elif column == "salary":
                column_definitions.append('"salary" REAL')
            elif df[column].dtype.kind in ("i", "f"):
                column_definitions.append(f'"{column}" REAL')
            else:
                column_definitions.append(f'"{column}" TEXT')
        column_definitions.append("password_hash TEXT NOT NULL")
        cur.execute(f"CREATE TABLE employees ({', '.join(column_definitions)})")

        insert_columns = [column for column in df.columns if column != "password"]
        insert_columns.append("password_hash")
        placeholders = ", ".join("?" for _ in insert_columns)
        insert_sql = (
            f'INSERT INTO employees ({", ".join(insert_columns)}) '
            f"VALUES ({placeholders})"
        )
        rows = []
        for _, row in df.iterrows():
            values = [row[column] for column in df.columns if column != "password"]
            values.append(hash_password(str(row["password"])))
            rows.append(values)

        cur.executemany(insert_sql, rows)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emp_id ON employees(id)")
        conn.commit()

    ensure_transactional_tables()
    print(f"Loaded {len(df)} employees into SQLite: {DB_PATH}")


def get_employee(employee_id: str) -> dict | None:
    """Fetch an employee profile without its password hash."""
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
    if row is None:
        return None
    employee = dict(row)
    employee.pop("password_hash", None)
    return employee
