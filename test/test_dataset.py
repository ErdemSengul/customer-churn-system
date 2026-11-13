"""
customers.csv veri seti testleri (churn tahmini + kampanya projesi).

Kontrol edilenler:
  1. Şema        : sütunlar, tipler, ID, kategorik değerler, hedef (churn)
  2. Eksik veri  : eksik veri VAR mı, sadece beklenen sütunlarda mı, oranı makul mü
  3. Aykırı değer: aykırı değer VAR mı, oranı makul mü, imkânsız değer var mı
  4. Tutarlılık  : sütunlar arası mantık kuralları (sipariş sayıları, oranlar,
                   tarihler, kampanya hunisi, sadakat segmenti, ...)

Çalıştırma (proje kök klasöründe):
    pytest -v src/data_gen/test_dataset.py

Veri dosyası şu sırayla aranır:
    1. CUSTOMERS_CSV ortam değişkeni
    2. data/customers.csv, data/raw/customers.csv, customers.csv
       (proje kökü, mevcut klasör ve bu dosyanın üst klasörleri)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
PRODUCTS = [
    "tisort", "gomlek_bluz", "kazak", "hirka", "sweatshirt", "pantolon", "jean",
    "elbise", "etek", "mont_kaban", "ceket_blazer", "sort_bermuda",
    "atki_bere_eldiven", "spor_giyim", "ic_giyim_corap",
]
SHARE_PRODUCT_COLS = [f"share_{p}" for p in PRODUCTS]
SHARE_CHANNEL_COLS = ["share_app", "share_web", "share_store"]
WINTER_COLS = [f"winter_prob_{p}" for p in PRODUCTS]

EXPECTED_COLUMNS = [
    "customer_id", "age", "gender", "city", "region", "city_cold_index",
    "income_level", "payment_method", "tenure_months", "first_purchase_date",
    "last_purchase_date", "recency_days", "order_count", "orders_last_12m",
    "orders_last_3m", "orders_prev_3m", "avg_days_between_orders",
    "recency_ratio", "total_spent", "spend_last_12m", "avg_basket",
    "discount_order_ratio", "return_count", "return_rate", "complaint_count",
    "support_tickets", "csat", "nps", "app_sessions_per_month",
    "web_visits_per_month", *SHARE_CHANNEL_COLS, "preferred_product",
    *SHARE_PRODUCT_COLS, "campaigns_received", "campaign_opens",
    "campaign_clicks", "campaign_conversions", "email_open_rate",
    "email_click_rate", "campaign_conversion_rate", "email_unsubscribed",
    "loyalty_member", "loyalty_tier", "points_balance", "referral_count",
    "expected_winter_orders", *WINTER_COLS, "top_winter_product",
    "loyalty_score", "loyalty_segment", "churn",
]

CATEGORIES = {
    "gender": {"kadin", "erkek"},
    "income_level": {"dusuk", "orta", "yuksek"},
    "payment_method": {"kredi_karti", "havale", "kapida_odeme", "dijital_cuzdan"},
    "region": {"Marmara", "Ege", "Akdeniz", "İç Anadolu", "Karadeniz",
               "Doğu Anadolu", "Güneydoğu Anadolu"},
    "loyalty_tier": {"yok", "bronz", "gumus", "altin"},
    "loyalty_segment": {"Kayip_Riski", "Riskli", "Potansiyel", "Sadik", "Sampiyon"},
    "preferred_product": set(PRODUCTS),
    "top_winter_product": set(PRODUCTS),
}

# Eksik veri olması BEKLENEN sütunlar
MISSING_EXPECTED = [
    "age", "avg_basket", "csat", "nps",
    "app_sessions_per_month", "web_visits_per_month",
]
MAX_MISSING_RATE = 0.10      # sütun başına en fazla %10 eksik
MAX_MISSING_PER_ROW = 3      # bir satırda en fazla 3 eksik hücre

# Aykırı değer olması BEKLENEN sütunlar (sağa çarpık davranış verileri)
OUTLIER_EXPECTED = [
    "order_count", "orders_last_3m", "total_spent", "avg_basket",
    "return_count", "support_tickets", "app_sessions_per_month",
    "web_visits_per_month",
]
EXTREME_K = 3.0              # uç aykırı: Q3 + 3*IQR üstü
MAX_EXTREME_RATE = 0.10      # sütun başına en fazla %10 uç aykırı

# Geçerli aralıklar: aykırı değer olabilir ama imkânsız değer olamaz
RANGES = {
    "age": (18, 100),
    "city_cold_index": (0, 1),
    "tenure_months": (0, 600),
    "recency_days": (0, None),
    "order_count": (1, None),
    "total_spent": (0, None),
    "spend_last_12m": (0, None),
    "avg_basket": (0, None),
    "csat": (1, 5),
    "nps": (0, 10),
    "loyalty_score": (0, 100),
    "points_balance": (0, None),
    **{c: (0, 1) for c in [
        "discount_order_ratio", "return_rate", "email_open_rate",
        "email_click_rate", "campaign_conversion_rate",
        *SHARE_CHANNEL_COLS, *SHARE_PRODUCT_COLS, *WINTER_COLS,
    ]},
    **{c: (0, None) for c in [
        "orders_last_12m", "orders_last_3m", "orders_prev_3m", "return_count",
        "complaint_count", "support_tickets", "app_sessions_per_month",
        "web_visits_per_month", "campaigns_received", "campaign_opens",
        "campaign_clicks", "campaign_conversions", "referral_count",
        "expected_winter_orders", "avg_days_between_orders", "recency_ratio",
    ]},
}

# Sadakat skoruna göre segment sınırları: (alt, üst]
SEGMENT_BINS = {
    "Kayip_Riski": (-np.inf, 35),
    "Riskli": (35, 50),
    "Potansiyel": (50, 65),
    "Sadik": (65, 80),
    "Sampiyon": (80, np.inf),
}
# Sadakat arttıkça churn azalmalı
SEGMENT_ORDER = ["Kayip_Riski", "Riskli", "Potansiyel", "Sadik", "Sampiyon"]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _find_csv() -> Path | None:
    env = os.environ.get("CUSTOMERS_CSV")
    if env:
        return Path(env)
    names = ["data/customers.csv", "data/raw/customers.csv", "customers.csv"]
    roots = [Path.cwd(), *Path(__file__).resolve().parents]
    for root in roots:
        for n in names:
            p = root / n
            if p.is_file():
                return p
    return None


def _bad_ids(df: pd.DataFrame, mask: pd.Series, n: int = 5) -> str:
    """Hata mesajı için kuralı bozan satırların sayısı ve ilk birkaç ID'si."""
    ids = df.loc[mask, "customer_id"].head(n).tolist()
    return f"{int(mask.sum())} satır, örn. {ids}"


def _iqr_upper(s: pd.Series, k: float) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return q3 + k * (q3 - q1)


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    path = _find_csv()
    if path is None or not path.is_file():
        pytest.fail(
            "customers.csv bulunamadı. data/customers.csv konumuna koyun "
            "veya CUSTOMERS_CSV ortam değişkeniyle yolunu verin."
        )
    return pd.read_csv(path)


# ===========================================================================
# 1. Şema
# ===========================================================================
def test_columns(df):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    assert not missing, f"Eksik sütunlar: {missing}"
    assert not extra, f"Beklenmeyen sütunlar: {extra}"


def test_row_count(df):
    assert len(df) >= 1000, f"Veri seti çok küçük: {len(df)} satır"


def test_customer_id(df):
    assert df["customer_id"].notna().all()
    assert df["customer_id"].is_unique, "Tekrarlanan customer_id var"
    assert df["customer_id"].astype(str).str.fullmatch(r"C\d{5,}").all(), \
        "customer_id formatı 'C00001' gibi olmalı"


def test_no_duplicate_rows(df):
    assert not df.drop(columns="customer_id").duplicated().any(), \
        "ID dışında birebir aynı satırlar var"


def test_numeric_dtypes(df):
    bad = [c for c in RANGES if not pd.api.types.is_numeric_dtype(df[c])]
    assert not bad, f"Sayısal olması gereken sütunlar: {bad}"


@pytest.mark.parametrize("col", sorted(CATEGORIES))
def test_categories(df, col):
    found = set(df[col].dropna().unique())
    unknown = found - CATEGORIES[col]
    assert not unknown, f"{col} sütununda tanımsız değerler: {unknown}"


def test_binary_columns(df):
    for col in ["churn", "loyalty_member", "email_unsubscribed"]:
        assert set(df[col].unique()) <= {0, 1}, f"{col} sadece 0/1 olmalı"


def test_churn_target(df):
    assert df["churn"].notna().all(), "Hedef değişken churn'de eksik olamaz"
    rate = df["churn"].mean()
    assert 0.05 <= rate <= 0.50, f"Churn oranı beklenmedik: %{rate*100:.1f}"


def test_dates_parse(df):
    for col in ["first_purchase_date", "last_purchase_date"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        assert parsed.notna().all(), f"{col} içinde okunamayan tarih var"


# ===========================================================================
# 2. Eksik veri
# ===========================================================================
def test_dataset_has_missing_values(df):
    """Proje gereği veri setinde eksik veri BULUNMALI."""
    assert df.isna().sum().sum() > 0, "Veri setinde hiç eksik veri yok"


@pytest.mark.parametrize("col", MISSING_EXPECTED)
def test_expected_columns_have_missing(df, col):
    assert df[col].isna().any(), f"{col} sütununda eksik veri bekleniyordu"


@pytest.mark.parametrize("col", MISSING_EXPECTED)
def test_missing_rate_reasonable(df, col):
    rate = df[col].isna().mean()
    assert rate <= MAX_MISSING_RATE, f"{col}: eksik oranı çok yüksek (%{rate*100:.1f})"


def test_missing_only_in_expected_columns(df):
    na = df.isna().sum()
    unexpected = na[(na > 0) & ~na.index.isin(MISSING_EXPECTED)]
    assert unexpected.empty, f"Eksik olmaması gereken sütunlarda eksik var: {unexpected.to_dict()}"


def test_critical_columns_complete(df):
    for col in ["customer_id", "churn", "loyalty_score", "loyalty_segment",
                "order_count", "total_spent", "recency_days"]:
        assert df[col].notna().all(), f"{col} kritik sütun, eksik olamaz"


def test_missing_per_row(df):
    per_row = df.isna().sum(axis=1)
    mask = per_row > MAX_MISSING_PER_ROW
    assert not mask.any(), f"Çok fazla eksiği olan satırlar: {_bad_ids(df, mask)}"


@pytest.mark.parametrize("col", MISSING_EXPECTED)
def test_missing_not_strongly_tied_to_churn(df, col):
    """Eksiklik hedefle güçlü ilişkiliyse (sızıntı) model eksikliği 'öğrenir'."""
    m = df[col].isna()
    if m.sum() < 30:
        pytest.skip("karşılaştırma için çok az eksik")
    diff = abs(df.loc[m, "churn"].mean() - df.loc[~m, "churn"].mean())
    assert diff < 0.15, f"{col}: eksik/dolu satırların churn oranı farkı %{diff*100:.1f}"


# ===========================================================================
# 3. Aykırı değer
# ===========================================================================
def test_dataset_has_outliers(df):
    """Proje gereği veri setinde aykırı değer BULUNMALI."""
    total = sum(
        (df[c] > _iqr_upper(df[c].dropna(), EXTREME_K)).sum() for c in OUTLIER_EXPECTED
    )
    assert total > 0, "Veri setinde hiç uç aykırı değer yok"


@pytest.mark.parametrize("col", OUTLIER_EXPECTED)
def test_expected_columns_have_outliers(df, col):
    s = df[col].dropna()
    n = int((s > _iqr_upper(s, EXTREME_K)).sum())
    assert n > 0, f"{col}: Q3 + {EXTREME_K}*IQR üstünde değer yok"


@pytest.mark.parametrize("col", OUTLIER_EXPECTED)
def test_outlier_rate_reasonable(df, col):
    s = df[col].dropna()
    rate = (s > _iqr_upper(s, EXTREME_K)).mean()
    assert rate <= MAX_EXTREME_RATE, f"{col}: uç aykırı oranı çok yüksek (%{rate*100:.1f})"


@pytest.mark.parametrize("col", OUTLIER_EXPECTED)
def test_outliers_are_far_from_bulk(df, col):
    """En büyük değer, %99'luk dilimin belirgin şekilde üstünde olmalı."""
    s = df[col].dropna()
    assert s.max() > 1.5 * s.quantile(0.99), f"{col}: belirgin uç değer yok"


@pytest.mark.parametrize("col", sorted(RANGES))
def test_value_ranges(df, col):
    """Aykırı değerler olabilir ama imkânsız değerler (negatif adet vb.) olamaz."""
    lo, hi = RANGES[col]
    s = df[col]
    mask = pd.Series(False, index=df.index)
    if lo is not None:
        mask |= s < lo
    if hi is not None:
        mask |= s > hi
    assert not mask.any(), f"{col} [{lo}, {hi}] dışında: {_bad_ids(df, mask)}"


# ===========================================================================
# 4. Tutarlılık kuralları
# ===========================================================================
def test_product_shares_sum_to_one(df):
    s = df[SHARE_PRODUCT_COLS].sum(axis=1)
    mask = (s - 1).abs() > 0.01
    assert not mask.any(), f"Ürün payları toplamı 1 değil: {_bad_ids(df, mask)}"


def test_channel_shares_sum_to_one(df):
    s = df[SHARE_CHANNEL_COLS].sum(axis=1)
    mask = (s - 1).abs() > 0.01
    assert not mask.any(), f"Kanal payları toplamı 1 değil: {_bad_ids(df, mask)}"


def test_preferred_product_is_max_share(df):
    mx = df[SHARE_PRODUCT_COLS].max(axis=1)
    pref = df.apply(lambda r: r[f"share_{r['preferred_product']}"], axis=1)
    mask = (mx - pref) > 1e-9
    assert not mask.any(), f"preferred_product en yüksek paylı ürün değil: {_bad_ids(df, mask)}"


def test_top_winter_product_is_max_prob(df):
    mx = df[WINTER_COLS].max(axis=1)
    top = df.apply(lambda r: r[f"winter_prob_{r['top_winter_product']}"], axis=1)
    mask = (mx - top) > 1e-9
    assert not mask.any(), f"top_winter_product en olası ürün değil: {_bad_ids(df, mask)}"


def test_order_counts_nested(df):
    mask = (
        (df["orders_last_3m"] + df["orders_prev_3m"] > df["orders_last_12m"])
        | (df["orders_last_12m"] > df["order_count"])
    )
    assert not mask.any(), f"Sipariş sayıları tutarsız: {_bad_ids(df, mask)}"


@pytest.mark.xfail(
    reason="BİLİNEN VERİ HATASI: bazı müşterilerde son 12 ay harcaması toplam "
           "harcamadan büyük (generate_synthetic.py düzeltilmeli)",
    strict=False,
)
def test_spend_last_12m_not_above_total(df):
    mask = df["spend_last_12m"] > df["total_spent"] + 0.01
    assert not mask.any(), f"spend_last_12m > total_spent: {_bad_ids(df, mask)}"


def test_returns(df):
    mask = df["return_count"] > df["order_count"]
    assert not mask.any(), f"İade sayısı sipariş sayısından fazla: {_bad_ids(df, mask)}"
    expected = df["return_count"] / df["order_count"]
    mask = (df["return_rate"] - expected).abs() > 1e-3
    assert not mask.any(), f"return_rate yanlış hesaplanmış: {_bad_ids(df, mask)}"


def test_campaign_funnel(df):
    mask = (
        (df["campaign_opens"] > df["campaigns_received"])
        | (df["campaign_clicks"] > df["campaign_opens"])
        | (df["campaign_conversions"] > df["campaign_clicks"])
    )
    assert not mask.any(), f"Kampanya hunisi tutarsız (alındı ≥ açıldı ≥ tıklandı ≥ dönüştü): {_bad_ids(df, mask)}"


def test_purchase_dates_order(df):
    first = pd.to_datetime(df["first_purchase_date"])
    last = pd.to_datetime(df["last_purchase_date"])
    mask = first > last
    assert not mask.any(), f"İlk alışveriş son alışverişten sonra: {_bad_ids(df, mask)}"


def test_recency_matches_single_reference_date(df):
    """Tüm müşteriler için last_purchase_date + recency_days aynı gün olmalı."""
    ref = pd.to_datetime(df["last_purchase_date"]) + pd.to_timedelta(df["recency_days"], unit="D")
    assert ref.nunique() == 1, f"recency_days farklı referans tarihleri veriyor: {ref.value_counts().head().to_dict()}"


def test_tenure_matches_first_purchase(df):
    ref = (pd.to_datetime(df["last_purchase_date"])
           + pd.to_timedelta(df["recency_days"], unit="D")).max()
    months = (ref - pd.to_datetime(df["first_purchase_date"])).dt.days / 30.44
    mask = (months - df["tenure_months"]).abs() > 2
    assert not mask.any(), f"tenure_months ile ilk alışveriş tarihi uyuşmuyor: {_bad_ids(df, mask)}"


def test_recency_ratio(df):
    expected = df["recency_days"] / df["avg_days_between_orders"]
    rel = (df["recency_ratio"] - expected).abs() / expected.clip(lower=0.01)
    mask = rel > 0.05
    assert not mask.any(), f"recency_ratio yanlış hesaplanmış: {_bad_ids(df, mask)}"


def test_loyalty_membership(df):
    mask = (df["loyalty_member"] == 0) != (df["loyalty_tier"] == "yok")
    assert not mask.any(), f"Üyelik ile seviye uyuşmuyor: {_bad_ids(df, mask)}"
    mask = (df["loyalty_member"] == 0) & (df["points_balance"] > 0)
    assert not mask.any(), f"Üye olmayan müşteride puan var: {_bad_ids(df, mask)}"


def test_loyalty_segment_matches_score(df):
    for seg, (lo, hi) in SEGMENT_BINS.items():
        s = df.loc[df["loyalty_segment"] == seg, "loyalty_score"]
        bad = s[(s <= lo) | (s > hi)]
        assert bad.empty, f"{seg} segmentinde aralık ({lo}, {hi}] dışı skor: {len(bad)} satır"


def test_churn_decreases_with_loyalty(df):
    """Projenin temel varsayımı: sadakat arttıkça churn oranı düşmeli."""
    rates = df.groupby("loyalty_segment")["churn"].mean().reindex(SEGMENT_ORDER)
    assert rates.notna().all(), f"Bazı segmentler veri setinde yok: {rates.to_dict()}"
    assert rates.is_monotonic_decreasing, f"Churn oranları sadakatle düşmüyor: {rates.round(3).to_dict()}"
