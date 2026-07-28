#!/usr/bin/env python3
"""Create the production database with normalized schema."""
import sqlite3
import os

os.makedirs("/app/data", exist_ok=True)
conn = sqlite3.connect("/app/data/production.db")
c = conn.cursor()

c.execute("""
CREATE TABLE steel_grades (
    grade_id INTEGER PRIMARY KEY,
    grade_code TEXT UNIQUE NOT NULL,
    grade_family TEXT NOT NULL
)
""")

grades = [
    (1, "LC-A", "low-carbon"),
    (2, "HC-B", "high-carbon"),
    (3, "SS-C", "stainless"),
    (4, "AL-D", "alloy"),
    (5, "TN-E", "tungsten"),
    (6, "NI-F", "nickel"),
    (7, "CR-G", "chromium"),
    (8, "MO-H", "molybdenum"),
    (9, "VN-I", "vanadium"),
    (10, "CB-J", "columbium"),
]
c.executemany("INSERT INTO steel_grades VALUES (?, ?, ?)", grades)

c.execute("""
CREATE TABLE work_orders (
    order_id INTEGER PRIMARY KEY,
    weight_kg INTEGER NOT NULL,
    steel_grade TEXT NOT NULL REFERENCES steel_grades(grade_code),
    release_window INTEGER NOT NULL,
    due_window INTEGER NOT NULL
)
""")

orders = [
    (0, 9, "LC-A", 0, 0),
    (1, 15, "HC-B", 0, 0),
    (2, 8, "SS-C", 0, 0),
    (3, 6, "AL-D", 0, 0),
    (4, 11, "TN-E", 0, 0),
    (5, 7, "NI-F", 0, 0),
    (6, 13, "CR-G", 0, 0),
    (7, 4, "MO-H", 0, 0),
    (8, 10, "LC-A", 0, 0),
    (9, 5, "HC-B", 0, 0),
    (10, 14, "VN-I", 0, 1),
    (11, 12, "CB-J", 0, 1),
    (12, 8, "SS-C", 0, 1),
    (13, 6, "AL-D", 1, 1),
    (14, 16, "TN-E", 1, 2),
    (15, 9, "NI-F", 1, 2),
    (16, 7, "CR-G", 1, 2),
    (17, 11, "MO-H", 1, 2),
    (18, 5, "VN-I", 2, 2),
    (19, 10, "CB-J", 2, 2),
]
c.executemany("INSERT INTO work_orders VALUES (?, ?, ?, ?, ?)", orders)

conn.commit()
conn.close()
print("Database created at /app/data/production.db")
