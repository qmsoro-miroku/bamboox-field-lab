
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "bamboox.db"


def init_database():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                species TEXT,
                origin TEXT,
                notes TEXT,
                filename TEXT,
                predicted_species TEXT,
                confidence_score REAL,
                inference_time_ms REAL,
                model_name TEXT,
                classification_type TEXT DEFAULT 'species',
                inspection_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()

    finally:
        connection.close()

    print("BambooX Database initialized!")
    print("Database location:", DATABASE_PATH)


if __name__ == "__main__":
    init_database()
