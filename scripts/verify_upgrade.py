"""Check generated employee data and database invariants."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_PATH, EXCEL_PATH
from src.database import TRANSACTIONAL_TABLES

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def main() -> None:
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    assert len(df) == 2000, f"Expected 2000 rows, got {len(df)}"
    print("[OK] 2000 rows")

    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values"
    print("[OK] 0 nulls")

    assert (df["paid_time_off_days"] >= 15).all(), "Some employees have PTO < 15"
    print(f"[OK] PTO min={df['paid_time_off_days'].min()}, max={df['paid_time_off_days'].max()}")

    month_cols = [f"absent_days_{month}" for month in MONTHS]
    max_absence = df[month_cols].max().max()
    assert max_absence <= 22, f"Max monthly absence = {max_absence}"
    print(f"[OK] Max monthly absence = {max_absence}")

    department_counts = df["department"].value_counts(normalize=True)
    targets = {
        "Engineering": 0.28, "Sales": 0.16, "Product": 0.10,
        "Operations": 0.10, "Customer Support": 0.10,
        "Marketing": 0.08, "Research & Development": 0.07,
        "Finance": 0.05, "Human Resources": 0.03, "Legal": 0.03,
    }
    for department, target in targets.items():
        actual = department_counts.get(department, 0)
        assert abs(actual - target) < 0.08, f"{department}: actual={actual:.3f} target={target:.3f}"
    print("[OK] Department weights within tolerance")

    with closing(sqlite3.connect(DB_PATH)) as conn:
        hashes = [row[0] for row in conn.execute("SELECT password_hash FROM employees").fetchall()]
        assert len(hashes) == 2000
        assert len({value[7:29] for value in hashes}) == 2000, "Employee bcrypt salts are not unique"
        print("[OK] 2000 unique bcrypt salts")

        employee_schema = conn.execute("PRAGMA table_info(employees)").fetchall()
        employee_columns = [row[1] for row in employee_schema]
        assert all(column in employee_columns for column in [
            "hire_date", "work_mode", "office_location", "health_plan", "performance_rating",
        ])
        print("[OK] All 5 new columns present in SQLite")

        column_types = {row[1]: row[2] for row in employee_schema}
        for column in ["years_of_expertise", "performance_rating", "sick_leave_days", "paid_time_off_days"]:
            assert column_types.get(column) == "INTEGER", f"{column} is {column_types.get(column)}, expected INTEGER"
        print("[OK] Integer columns stored as INTEGER")

        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table in TRANSACTIONAL_TABLES:
            assert table in tables, f"Missing table: {table}"
            row_count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            employee_count = conn.execute(
                f"SELECT count(DISTINCT employee_id) FROM {table}"
            ).fetchone()[0]
            missing_count = conn.execute(
                f"SELECT count(*) FROM employees e LEFT JOIN {table} t "
                "ON t.employee_id = e.id WHERE t.employee_id IS NULL"
            ).fetchone()[0]
            assert row_count > 0 and employee_count == 2000 and missing_count == 0, (
                f"{table}: {row_count} rows, {employee_count} employees, "
                f"{missing_count} employees without rows"
            )
            print(f"[OK] {table} has {row_count} rows for all 2000 employees")

    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
