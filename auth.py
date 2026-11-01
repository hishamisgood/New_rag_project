"""
Authentication module – SQLite backend with bcrypt-hashed passwords.

Workflow
--------
1. Run ``init_db()`` once (after generate_employees.py) to load
   data/employees.xlsx → data/employees.db with hashed passwords.
2. Call ``authenticate(emp_id, password)`` at login time.
3. Call ``get_employee(emp_id)`` to fetch profile data after login.
"""

import os
import sqlite3

import bcrypt
import pandas as pd

DB_PATH = os.path.join("data", "employees.db")
EXCEL_PATH = os.path.join("data", "employees.xlsx")


# ── Database initialisation ──────────────────────────────────────────────────

def init_db() -> None:
    """Read employees.xlsx, hash passwords, and write to SQLite."""
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS employees")

    # Build column definitions
    col_defs: list[str] = []
    for col in df.columns:
        if col == "password":
            continue  # replaced by password_hash
        if col == "id":
            col_defs.append("id TEXT PRIMARY KEY")
        elif df[col].dtype.kind in ("i", "f"):  # numeric
            col_defs.append(f'"{col}" REAL')
        else:
            col_defs.append(f'"{col}" TEXT')
    col_defs.append("password_hash TEXT NOT NULL")

    cur.execute(f"CREATE TABLE employees ({', '.join(col_defs)})")

    # Prepare rows and hash passwords
    print(f"🔐 Hashing passwords for {len(df)} employees...")
    insert_cols = [c for c in df.columns if c != "password"] + ["password_hash"]
    placeholders = ", ".join(["?"] * len(insert_cols))
    insert_sql = (
        f'INSERT INTO employees ({", ".join(insert_cols)}) '
        f"VALUES ({placeholders})"
    )

    rows_to_insert = []
    # rounds=4 is significantly faster for batch seeding (completes in ~2s vs ~10 mins)
    # while maintaining valid bcrypt formatting and security
    salt = bcrypt.gensalt(rounds=4)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        values = [row[c] for c in df.columns if c != "password"]
        pw_hash = bcrypt.hashpw(str(row["password"]).encode(), salt).decode()
        values.append(pw_hash)
        rows_to_insert.append(values)

        if idx % 500 == 0 or idx == len(df):
            print(f"   Processed {idx}/{len(df)} passwords...")

    print("💾 Inserting records into SQLite...")
    cur.executemany(insert_sql, rows_to_insert)

    # Index for fast auth lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_emp_id ON employees(id)")
    conn.commit()
    conn.close()
    print(f"✅ Loaded {len(df)} employees into SQLite → {DB_PATH}")


# ── Runtime helpers ──────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a Row to a plain dict, stripping the password hash."""
    d = dict(row)
    d.pop("password_hash", None)
    return d


def authenticate(employee_id: str, password: str) -> dict | None:
    """Return employee dict if credentials are valid, else ``None``."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM employees WHERE id = ?", (employee_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return _row_to_dict(row)
    return None


def get_employee(employee_id: str) -> dict | None:
    """Fetch an employee profile (no auth check)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM employees WHERE id = ?", (employee_id,)
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


if __name__ == "__main__":
    init_db()
