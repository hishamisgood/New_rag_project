"""Password hashing and employee login verification."""

import sqlite3
from contextlib import closing

import bcrypt

from .config import DB_PATH


def hash_password(password: str) -> str:
    """Return a bcrypt hash with a fresh salt."""
    salt = bcrypt.gensalt(rounds=4)
    return bcrypt.hashpw(password.encode(), salt).decode()


def authenticate(employee_id: str, password: str) -> dict | None:
    """Return an employee profile when the supplied credentials are valid."""
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()

    if row is None or not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return None

    employee = dict(row)
    employee.pop("password_hash", None)
    return employee
