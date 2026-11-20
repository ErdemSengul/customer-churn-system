"""
Proje genelinde kullanılan klasör yolları, sabitler ve veritabanı bağlantı ayarları.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# --- Klasörler ve Dosya Yolları ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

RAW_CSV = RAW_DIR / "customers.csv"
FEATURES_CSV = PROCESSED_DIR / "customers_features.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_churn_pipeline.joblib"
METRICS_CSV_PATH = PROCESSED_DIR / "cv_model_comparison.csv"
MLFLOW_DB_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

SQL_DIR = ROOT / "sql"
SCHEMA_SQL = SQL_DIR / "schema.sql"
CHECKS_SQL = SQL_DIR / "quality_checks.sql"

# --- .env dosyasını oku ---
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# --- PostgreSQL bağlantısı ---
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "churn"),
    "user": os.getenv("POSTGRES_USER", "churn"),
    "password": os.getenv("POSTGRES_PASSWORD", "churn"),
}

DATASET_VIEW = "churn_dataset"