# db.py
import sqlite3
import os

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "fitness.db")


def init_db():
    # ensure data dir exists
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        workout_type TEXT,
        workout_minutes REAL,
        calories_intake REAL,
        sleep_hours REAL,
        mood_score INTEGER,
        mood_text TEXT
    )
    """)
    conn.commit()
    conn.close()


def insert_log(
    date,
    workout_type,
    workout_minutes,
    calories_intake,
    sleep_hours,
    mood_score,
    mood_text,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO logs (date, workout_type, workout_minutes, calories_intake, sleep_hours, mood_score, mood_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            date,
            workout_type,
            workout_minutes,
            calories_intake,
            sleep_hours,
            mood_score,
            mood_text,
        ),
    )
    conn.commit()
    conn.close()


def fetch_logs():
    import pandas as pd

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY date DESC", conn)
    conn.close()
    return df
