"""
Veritabanı katmanı (PostgreSQL). Sorumlu: Arkın

Ne yapar?
  - Veritabanında 8 tabloyu ve bunları birleştiren "churn_dataset" görünümünü kurar.
  - Üretilen müşteri verisini (tek büyük tablo) 8 kaynağa bölerek veritabanına yazar.
  - Birleşik veriyi geri okuyup pandas DataFrame olarak verir.
  - Veri kalite kontrollerini çalıştırır.

Komut satırından (proje kök klasöründe):
  python -m src.data_loader setup               # her şeyi kur (ilk sefer bunu çalıştırın)
  python -m src.data_loader generate --n 10000  # yeni veri üret, CSV'ye ve veritabanına yaz
  python -m src.data_loader load                # data/raw/customers.csv dosyasını veritabanına yükle
  python -m src.data_loader export              # veritabanındaki birleşik veriyi CSV'ye yaz
  python -m src.data_loader check               # kalite kontrolleri
  python -m src.data_loader reset               # tabloları silip boş olarak yeniden kur

Python kodundan (ekibin geri kalanı için):
  from src.data_loader import load_customers, run_query
  df = load_customers()                                    # birleşik veri seti
  ozet = run_query("SELECT region, avg(churn) FROM churn_dataset GROUP BY region")
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from src.config import CHECKS_SQL, DATASET_VIEW, DB_CONFIG, RAW_CSV, ROOT, SCHEMA_SQL

# Kaynak tablolar (yazma sırası önemli: önce customers, çünkü diğerleri ona bağlı)
TABLES = [
    "customers",
    "orders_summary",
    "customer_support",
    "digital_activity",
    "product_preferences",
    "campaigns",
    "season_forecast",
    "labels",
]


class DatabaseError(Exception):
    """Veritabanına bağlanma / yazma sırasında anlaşılır hata mesajı."""


# ----------------------------------------------------------------------
# Bağlantı
# ----------------------------------------------------------------------
def get_connection():
    """PostgreSQL bağlantısı açar. Ayarlar src/config.py -> DB_CONFIG (.env dosyasından)."""
    try:
        import psycopg
    except ImportError as e:
        raise DatabaseError(
            "psycopg kurulu değil. Şunu çalıştırın: pip install -r requirements.txt"
        ) from e
    try:
        return psycopg.connect(**DB_CONFIG, connect_timeout=5)
    except psycopg.OperationalError as e:
        cfg = {k: v for k, v in DB_CONFIG.items() if k != "password"}
        raise DatabaseError(
            f"Veritabanına bağlanılamadı ({cfg}).\n"
            "  - PostgreSQL açık mı?  Docker kullanıyorsanız: docker compose up -d\n"
            "  - .env dosyasındaki kullanıcı adı / şifre / port doğru mu?\n"
            f"  Asıl hata: {e}"
        ) from e


def _execute_file(conn, path: Path) -> None:
    conn.execute(path.read_text(encoding="utf-8"))


def table_columns(conn, table: str) -> list[tuple[str, str]]:
    """Tablonun (sütun adı, veri tipi) listesi, tablodaki sırasıyla."""
    rows = conn.execute(
        """SELECT column_name, data_type FROM information_schema.columns
           WHERE table_schema = current_schema() AND table_name = %s
           ORDER BY ordinal_position""",
        (table,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# ----------------------------------------------------------------------
# Kurulum
# ----------------------------------------------------------------------
def init_db(reset: bool = False) -> None:
    """Tabloları ve görünümü kurar. reset=True: önce hepsini siler (veriler gider!)."""
    with get_connection() as conn:
        if reset:
            conn.execute(f"DROP VIEW IF EXISTS {DATASET_VIEW}")
            for t in reversed(TABLES):
                conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        _execute_file(conn, SCHEMA_SQL)


def count_customers() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT count(*) FROM customers").fetchone()[0]


# ----------------------------------------------------------------------
# Yazma: tek büyük tablo -> 8 kaynak tablo
# ----------------------------------------------------------------------
_INTEGER_TYPES = {"integer", "smallint", "bigint"}


def save_customers(df: pd.DataFrame) -> int:
    """DataFrame'i 8 tabloya bölüp veritabanına yazar. Eski veriler silinir.
    generate_synthetic.generate_and_store() bu fonksiyonu çağırır.
    Döndürür: yazılan müşteri sayısı."""
    import psycopg

    init_db()
    try:
        return _save(df)
    except psycopg.Error as e:
        raise DatabaseError(
            "Veri veritabanı kurallarına uymuyor; hiçbir şey yazılmadı, eski veri duruyor.\n"
            f"  Asıl hata: {e}"
        ) from e


def _save(df: pd.DataFrame) -> int:
    with get_connection() as conn:  # 'with' bloğu: hata olursa hiçbir şey yazılmaz
        plan = {t: table_columns(conn, t) for t in TABLES}

        needed = {c for cols in plan.values() for c, _ in cols}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise DatabaseError(
                "Veride veritabanının beklediği sütunlar yok: " + ", ".join(missing)
                + "\nVeri eski bir üretici sürümünden olabilir. Yeniden üretin: "
                  "python -m src.data_loader generate"
            )
        extra = sorted(set(df.columns) - needed)
        if extra:
            print(f"Uyarı: şu sütunlar veritabanında yok, atlandı: {', '.join(extra)}")

        conn.execute("TRUNCATE customers CASCADE")  # bağlı tüm tabloları da boşaltır
        with conn.cursor() as cur:
            for table, cols in plan.items():
                part = df[[c for c, _ in cols]].copy()
                for c, dtype in cols:
                    if dtype in _INTEGER_TYPES:      # 43.0 -> 43 (eksik değer boş kalır)
                        part[c] = part[c].astype("Int64")
                buf = part.to_csv(index=False, header=False)
                names = ", ".join(c for c, _ in cols)
                with cur.copy(f"COPY {table} ({names}) FROM STDIN WITH (FORMAT csv)") as cp:
                    cp.write(buf)
        n = conn.execute("SELECT count(*) FROM customers").fetchone()[0]
    return n


def load_csv(path: Path = RAW_CSV) -> int:
    """CSV dosyasını veritabanına yükler."""
    path = Path(path)
    if not path.is_file():
        raise DatabaseError(f"Dosya bulunamadı: {path}")
    return save_customers(pd.read_csv(path))


# ----------------------------------------------------------------------
# Okuma
# ----------------------------------------------------------------------
def load_customers() -> pd.DataFrame:
    """Birleşik veri seti (churn_dataset görünümü), customer_id sırasıyla.
    Sütunlar ve tipler data/raw/customers.csv ile aynıdır."""
    buf = io.BytesIO()
    with get_connection() as conn, conn.cursor() as cur:
        sql = f"COPY (SELECT * FROM {DATASET_VIEW} ORDER BY customer_id) TO STDOUT WITH (FORMAT csv, HEADER)"
        with cur.copy(sql) as cp:
            for chunk in cp:
                buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)


def run_query(sql: str, params=None) -> pd.DataFrame:
    """Herhangi bir SELECT sorgusunu çalıştırıp DataFrame döndürür.
    Değişken kullanırken %s yazın:  run_query("... WHERE region = %s", ("Ege",))"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def export_csv(path: Path = RAW_CSV) -> Path:
    """Veritabanındaki birleşik veriyi CSV'ye yazar (veritabanı kuramayan ekip arkadaşları için)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    load_customers().to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# Kalite kontrolleri
# ----------------------------------------------------------------------
def run_checks() -> pd.DataFrame:
    """sql/quality_checks.sql içindeki kontrolleri çalıştırır. 'gecti' sütunu True olmalı."""
    df = run_query(CHECKS_SQL.read_text(encoding="utf-8"))
    df["gecti"] = df["hatali_kayit"] == 0
    return df


def summary() -> dict:
    """Birleşik verinin kısa özeti (müşteri sayısı, churn oranı, tarih)."""
    r = run_query(f"""
        SELECT count(*) AS musteri,
               round(avg(churn) * 100, 1)::float8 AS churn_yuzde,
               min(snapshot_date)::text AS referans_tarihi,
               round(avg(loyalty_score)::numeric, 1)::float8 AS ort_sadakat
        FROM {DATASET_VIEW}""").iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}


# ----------------------------------------------------------------------
# Komut satırı
# ----------------------------------------------------------------------
def _print_checks() -> bool:
    checks = run_checks()
    for _, r in checks.iterrows():
        mark = "OK  " if r.gecti else "HATA"
        tail = "" if r.gecti else f"  ({r.hatali_kayit} kayıt)"
        print(f"  [{mark}] {r.kontrol}{tail}")
    ok = bool(checks.gecti.all())
    print(f"  {int(checks.gecti.sum())}/{len(checks)} kontrol geçti.")
    return ok


def _generate(n: int, churn: float, seed: int) -> int:
    from src.generate_synthetic import GeneratorConfig, generate_and_store

    cfg = GeneratorConfig(n_customers=n, churn_target_rate=churn, seed=seed)
    df, report = generate_and_store(
        cfg, progress=lambda p, msg: print(f"  %{int(p * 100):3d}  {msg}")
    )
    print(f"  {len(df)} müşteri üretildi -> {RAW_CSV.relative_to(ROOT)} ve veritabanı")
    return len(df)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.data_loader", description="Churn veritabanı araçları")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="tabloları kur, veri yoksa yükle/üret, kontrol et")
    sub.add_parser("init", help="sadece tabloları kur")
    sub.add_parser("reset", help="tabloları silip boş kur (veriler gider)")
    g = sub.add_parser("generate", help="yeni veri üret ve yükle")
    g.add_argument("--n", type=int, default=10_000, help="müşteri sayısı (varsayılan 10000)")
    g.add_argument("--churn", type=float, default=0.25, help="hedef churn oranı (varsayılan 0.25)")
    g.add_argument("--seed", type=int, default=42)
    ld = sub.add_parser("load", help="CSV'yi veritabanına yükle")
    ld.add_argument("--csv", type=Path, default=RAW_CSV)
    ex = sub.add_parser("export", help="birleşik veriyi CSV'ye yaz")
    ex.add_argument("--out", type=Path, default=RAW_CSV)
    sub.add_parser("check", help="kalite kontrolleri")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "init":
            init_db()
            print("Tablolar hazır.")
        elif args.cmd == "reset":
            init_db(reset=True)
            print("Tablolar silindi ve boş olarak yeniden kuruldu.")
        elif args.cmd == "setup":
            print("1/3 Tablolar kuruluyor...")
            init_db()
            n = count_customers()
            if n:
                print(f"2/3 Veritabanında zaten {n} müşteri var, yükleme atlandı.")
            elif RAW_CSV.is_file():
                print(f"2/3 {RAW_CSV.name} veritabanına yükleniyor...")
                print(f"  {load_csv(RAW_CSV)} müşteri yüklendi.")
            else:
                print("2/3 CSV bulunamadı, yeni veri üretiliyor...")
                _generate(10_000, 0.25, 42)
            print("3/3 Kalite kontrolleri:")
            ok = _print_checks()
            print("Özet:", summary())
            return 0 if ok else 1
        elif args.cmd == "generate":
            _generate(args.n, args.churn, args.seed)
        elif args.cmd == "load":
            print(f"{load_csv(args.csv)} müşteri yüklendi.")
        elif args.cmd == "export":
            print(f"Yazıldı: {export_csv(args.out)}")
        elif args.cmd == "check":
            return 0 if _print_checks() else 1
    except DatabaseError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
