"""
Proje genelinde kullanılan klasör yolları ve veritabanı bağlantı ayarları.

Veritabanı ayarları proje kökündeki .env dosyasından okunur
(.env.example dosyasını kopyalayıp .env adıyla kaydedin).
.env yoksa aşağıdaki varsayılanlar kullanılır; bunlar docker-compose.yml ile uyumludur.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Klasörler ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "customers.csv"          # birleşik ham veri seti (CSV kopyası)
SQL_DIR = ROOT / "sql"
SCHEMA_SQL = SQL_DIR / "schema.sql"          # tablolar + churn_dataset görünümü
CHECKS_SQL = SQL_DIR / "quality_checks.sql"  # veri kalite kontrolleri

# --- .env dosyasını oku (python-dotenv kuruluysa) ---
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

# Ekibin okuyacağı birleşik tablo (görünüm) adı
DATASET_VIEW = "churn_dataset"
