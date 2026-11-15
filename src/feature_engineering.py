"""
RFM skorlama, müşteri segmentasyonu ve davranışsal özellik üretimi (Feature Engineering).

Kullanım (Komut satırından):
    python -m src.feature_engineering

Kullanım (Python kodundan - pipeline veya train içinden):
    from src.feature_engineering import build_features, load_or_build_features
    df_feat = build_features(df)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR, RAW_CSV

PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = PROCESSED_DIR / "customers_features.csv"

PRODUCTS = [
    "tisort", "gomlek_bluz", "kazak", "hirka", "sweatshirt", "pantolon", "jean",
    "elbise", "etek", "mont_kaban", "ceket_blazer", "sort_bermuda",
    "atki_bere_eldiven", "spor_giyim", "ic_giyim_corap",
]
SHARE_PRODUCT_COLS = [f"share_{p}" for p in PRODUCTS]


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency, Frequency ve Monetary sütunlarından 1-5 arası RFM skorlarını,
    ağırlıklı RFM puanını ve eyleme dönüştürülebilir RFM segmentlerini üretir.
    """
    data = df.copy()

    # R Skoru: Son alışverişten geçen gün sayısı (recency_days) düşükse puan yüksek (5)
    data["r_score"] = pd.qcut(
        data["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # F Skoru: Toplam sipariş sayısı (order_count) yüksekse puan yüksek (5)
    data["f_score"] = pd.qcut(
        data["order_count"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # M Skoru: Toplam harcama (total_spent) yüksekse puan yüksek (5)
    data["m_score"] = pd.qcut(
        data["total_spent"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Birleşik RFM Metrikleri
    data["rfm_total_score"] = data["r_score"] + data["f_score"] + data["m_score"]
    data["rfm_weighted_score"] = (
        0.40 * data["r_score"] + 0.35 * data["f_score"] + 0.25 * data["m_score"]
    ).round(2)

    # İşletme aksiyonu için RFM Segmentasyonu
    conditions = [
        (data["r_score"] >= 4) & (data["f_score"] >= 4) & (data["m_score"] >= 4),
        (data["r_score"] >= 3) & (data["f_score"] >= 3),
        (data["r_score"] <= 2) & (data["f_score"] >= 4),
        (data["r_score"] >= 4) & (data["f_score"] <= 2),
        (data["r_score"] <= 2) & (data["f_score"] <= 2),
    ]
    choices = [
        "Sampiyonlar",
        "Sadik_Musteriler",
        "Kaybedilmemesi_Gerekenler",
        "Yeni_Potansiyeller",
        "Uykuda_Riskli",
    ]
    data["rfm_segment"] = np.select(conditions, choices, default="Diger_Gelisime_Acik")

    return data


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Müşterinin sipariş ivmesini, harcama yoğunluğunu, memnuniyetsizlik/sürtünme
    endeksini ve çok kanallı (omnichannel) etkileşim davranışlarını türetir.
    """
    data = df.copy()
    safe_orders = np.maximum(data["order_count"], 1)
    safe_tenure = np.maximum(data["tenure_months"], 1)
    safe_spent = np.maximum(data["total_spent"], 1.0)

    # 1. Sipariş İvmesi (Son 3 ay vs Önceki 3 ay aktivite değişimi)
    data["order_momentum_diff"] = data["orders_last_3m"] - data["orders_prev_3m"]
    data["order_momentum_ratio"] = (
        (data["orders_last_3m"] + 1.0) / (data["orders_prev_3m"] + 1.0)
    ).round(4)

    # 2. Son 1 Yıl Aktivite ve Harcama Payı (Müşteri son yılda ne kadar aktif?)
    data["recent_12m_order_share"] = (data["orders_last_12m"] / safe_orders).round(4)
    data["recent_12m_spend_share"] = (data["spend_last_12m"] / safe_spent).round(4)

    # 3. Müşteri Yaşam Süresi (Tenure) Bazlı Hız Metrikleri
    data["orders_per_tenure_month"] = (data["order_count"] / safe_tenure).round(4)
    data["spend_per_tenure_month"] = (data["total_spent"] / safe_tenure).round(2)

    # 4. Memnuniyetsizlik ve Sürtünme Endeksi (İade + Şikayet + Destek Talebi)
    data["complaint_per_order"] = (data["complaint_count"] / safe_orders).round(4)
    data["support_ticket_per_order"] = (data["support_tickets"] / safe_orders).round(4)
    data["friction_index"] = (
        data["return_rate"]
        + (data["complaint_per_order"] * 2.0)
        + data["support_ticket_per_order"]
    ).round(4)

    # 5. Dijital Etkileşim ve Kanal Davranışı
    app_sess = data["app_sessions_per_month"].fillna(data["app_sessions_per_month"].median())
    web_vis = data["web_visits_per_month"].fillna(data["web_visits_per_month"].median())
    data["total_digital_touchpoints"] = (app_sess + web_vis).round(2)
    data["digital_channel_share"] = (data["share_app"] + data["share_web"]).round(4)

    # 6. Kampanya Duyarlılığı ve Ürün Çeşitliliği
    safe_campaigns = np.maximum(data["campaigns_received"], 1)
    data["campaign_engagement_score"] = (
        (data["campaign_opens"] * 0.3 + data["campaign_clicks"] * 0.7) / safe_campaigns
    ).round(4)

    existing_prod_cols = [c for c in SHARE_PRODUCT_COLS if c in data.columns]
    if existing_prod_cols:
        data["active_category_count"] = (data[existing_prod_cols] > 0.05).sum(axis=1)

    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hem RFM hem de davranışsal özellikleri sırasıyla uygular."""
    data = add_rfm_features(df)
    data = add_behavioral_features(data)
    return data


def load_or_build_features(
    raw_path: Path = RAW_CSV,
    output_path: Path = FEATURES_CSV,
    save: bool = True,
) -> pd.DataFrame:
    """Ham veriyi okur, özellikleri üretir ve data/processed/ altına kaydeder."""
    df_raw = pd.read_csv(raw_path)
    df_feat = build_features(df_raw)
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(output_path, index=False)
    return df_feat


if __name__ == "__main__":
    df_result = load_or_build_features()
    yeni_sutunlar = [
        "r_score", "f_score", "m_score", "rfm_total_score", "rfm_weighted_score",
        "rfm_segment", "order_momentum_diff", "order_momentum_ratio",
        "recent_12m_order_share", "recent_12m_spend_share",
        "orders_per_tenure_month", "spend_per_tenure_month",
        "complaint_per_order", "support_ticket_per_order", "friction_index",
        "total_digital_touchpoints", "digital_channel_share",
        "campaign_engagement_score", "active_category_count",
    ]
    print(f"[OK] Feature Engineering tamamlandi! Yeni tablo boyutu: {df_result.shape}")
    print(f"[OK] Uretilen yeni ozellik sayisi: {len(yeni_sutunlar)}")
    print(f"[OK] Dosya kaydedildi: {FEATURES_CSV}\n")
    print("--- RFM Segment Ozeti ---")
    summary = (
        df_result.groupby("rfm_segment")
        .agg(
            musteri_sayisi=("customer_id", "count"),
            churn_orani=("churn", "mean"),
            ort_recency=("recency_days", "mean"),
            ort_siparis=("order_count", "mean"),
            ort_toplam_harcama=("total_spent", "mean"),
        )
        .round(3)
        .sort_values(by="churn_orani", ascending=False)
    )
    print(summary.to_string())