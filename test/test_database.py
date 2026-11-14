"""
Veritabanı testleri (Arkın).

Çalıştırma (proje kök klasöründe, PostgreSQL açıkken):
    pytest -v test/test_database.py

PostgreSQL kapalıysa testler hata vermez, "atlandı" (skipped) olarak görünür.
"""
import pytest

from src import data_loader as dl
from src.generate_synthetic import GeneratorConfig, generate


@pytest.fixture(scope="module")
def conn_ok():
    try:
        dl.init_db()
    except dl.DatabaseError as e:
        pytest.skip(f"Veritabanına bağlanılamadı: {e}")
    if dl.count_customers() == 0:
        pytest.skip("Veritabanı boş. Önce: python -m src.data_loader setup")
    return True


def test_tablolar_var(conn_ok):
    with dl.get_connection() as conn:
        for t in dl.TABLES:
            assert dl.table_columns(conn, t), f"{t} tablosu yok"


def test_birlesik_veri_sutunlari_uretici_ile_ayni(conn_ok):
    """Üretici (generate_synthetic.py) değişirse schema.sql de güncellenmeli."""
    uretilen = generate(GeneratorConfig(n_customers=1_000))
    with dl.get_connection() as conn:
        gorunum = [c for c, _ in dl.table_columns(conn, "churn_dataset")]
    assert gorunum == list(uretilen.columns)


def test_her_musteri_tek_satir(conn_ok):
    df = dl.load_customers()
    assert len(df) == dl.count_customers()
    assert df["customer_id"].is_unique


def test_kalite_kontrolleri(conn_ok):
    checks = dl.run_checks()
    hatali = checks[~checks["gecti"]]
    assert hatali.empty, "Başarısız kontroller:\n" + hatali.to_string(index=False)
