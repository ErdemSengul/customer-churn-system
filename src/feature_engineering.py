"""
RFM skorlama, müşteri segmentasyonu ve davranışsal özellik üretimi (Feature Engineering).
Hem Scikit-Learn Pipeline içinde (sızıntısız fit/transform) hem de standalone kullanılabilir.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import FEATURES_CSV, RAW_CSV

PRODUCTS = [
    "tisort", "gomlek_bluz", "kazak", "hirka", "sweatshirt", "pantolon", "jean",
    "elbise", "etek", "mont_kaban", "ceket_blazer", "sort_bermuda",
    "atki_bere_eldiven", "spor_giyim", "ic_giyim_corap",
]
SHARE_PRODUCT_COLS = [f"share_{p}" for p in PRODUCTS]

ENGINEERED_NUM_COLS = [
    "r_score", "f_score", "m_score", "rfm_total_score", "rfm_weighted_score",
    "order_momentum_diff", "order_momentum_ratio",
    "recent_12m_order_share", "recent_12m_spend_share",
    "orders_per_tenure_month", "spend_per_tenure_month",
    "complaint_per_order", "support_ticket_per_order", "friction_index",
    "total_digital_touchpoints", "digital_channel_share",
    "campaign_engagement_score", "active_category_count",
]
ENGINEERED_CAT_COLS = ["rfm_segment"]


def _compute_quantile_bins(series: pd.Series, q: int = 5) -> np.ndarray:
    """Eğitim setinden kesikli dilim sınırlarını üretir; tekrarlı değerleri güvenli birleştirir."""
    clean = series.dropna()
    if clean.empty:
        return np.array([-np.inf, 1, 2, 3, 4, np.inf])
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.unique(np.quantile(clean, quantiles))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _apply_score_bins(series: pd.Series, bins: np.ndarray, reverse: bool = False) -> pd.Series:
    """Öğrenilmiş sınırlarla 1-5 arası RFM skoru atar (tek satırlık veride de çalışır)."""
    n_intervals = len(bins) - 1
    labels = list(range(n_intervals, 0, -1)) if reverse else list(range(1, n_intervals + 1))
    binned = pd.cut(series.fillna(series.median() if series.notna().any() else 0),
                    bins=bins, labels=labels, include_lowest=True)
    return binned.astype(int)


def add_rfm_features(
    df: pd.DataFrame,
    r_bins: np.ndarray | None = None,
    f_bins: np.ndarray | None = None,
    m_bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Recency, Frequency ve Monetary sütunlarından 1-5 RFM skorları ve segment üretir."""
    data = df.copy()

    r_edges = r_bins if r_bins is not None else _compute_quantile_bins(data["recency_days"], q=5)
    f_edges = f_bins if f_bins is not None else _compute_quantile_bins(data["order_count"], q=5)
    m_edges = m_bins if m_bins is not None else _compute_quantile_bins(data["total_spent"], q=5)

    data["r_score"] = _apply_score_bins(data["recency_days"], r_edges, reverse=True)
    data["f_score"] = _apply_score_bins(data["order_count"], f_edges, reverse=False)
    data["m_score"] = _apply_score_bins(data["total_spent"], m_edges, reverse=False)

    data["rfm_total_score"] = data["r_score"] + data["f_score"] + data["m_score"]
    data["rfm_weighted_score"] = (
        0.40 * data["r_score"] + 0.35 * data["f_score"] + 0.25 * data["m_score"]
    ).round(2)

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
    """Sipariş ivmesi, sürtünme endeksi, dijital temas ve kategori çeşitliliği türetir."""
    data = df.copy()
    safe_orders = np.maximum(data["order_count"], 1)
    safe_tenure = np.maximum(data["tenure_months"], 1)
    safe_spent = np.maximum(data["total_spent"], 1.0)
    safe_campaigns = np.maximum(data["campaigns_received"], 1)

    # 1. Sipariş İvmesi
    data["order_momentum_diff"] = data["orders_last_3m"] - data["orders_prev_3m"]
    data["order_momentum_ratio"] = (
        (data["orders_last_3m"] + 1.0) / (data["orders_prev_3m"] + 1.0)
    ).round(4)

    # 2. Son 1 Yıl Aktivite ve Harcama Payı
    data["recent_12m_order_share"] = (data["orders_last_12m"] / safe_orders).round(4)
    data["recent_12m_spend_share"] = (data["spend_last_12m"] / safe_spent).round(4)

    # 3. Müşteri Yaşam Süresi Bazlı Hız Metrikleri
    data["orders_per_tenure_month"] = (data["order_count"] / safe_tenure).round(4)
    data["spend_per_tenure_month"] = (data["total_spent"] / safe_tenure).round(2)

    # 4. Memnuniyetsizlik ve Sürtünme Endeksi
    data["complaint_per_order"] = (data["complaint_count"] / safe_orders).round(4)
    data["support_ticket_per_order"] = (data["support_tickets"] / safe_orders).round(4)
    data["friction_index"] = (
        data["return_rate"]
        + (data["complaint_per_order"] * 2.0)
        + data["support_ticket_per_order"]
    ).round(4)

    # 5. Dijital Etkileşim (eksikler SimpleImputer tarafından doldurulmak üzere korunur)
    data["total_digital_touchpoints"] = (
        data["app_sessions_per_month"] + data["web_visits_per_month"]
    ).round(2)
    data["digital_channel_share"] = (data["share_app"] + data["share_web"]).round(4)

    # 6. Kampanya Duyarlılığı ve Ürün Çeşitliliği
    data["campaign_engagement_score"] = (
        (data["campaign_opens"] * 0.3 + data["campaign_clicks"] * 0.7) / safe_campaigns
    ).round(4)

    existing_prod_cols = [c for c in SHARE_PRODUCT_COLS if c in data.columns]
    if existing_prod_cols:
        data["active_category_count"] = (data[existing_prod_cols] > 0.05).sum(axis=1)
    else:
        data["active_category_count"] = 0

    return data


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn uyumlu sızıntısız özellik üretici."""

    def fit(self, X: pd.DataFrame, y=None):
        self.r_bins_ = _compute_quantile_bins(X["recency_days"], q=5)
        self.f_bins_ = _compute_quantile_bins(X["order_count"], q=5)
        self.m_bins_ = _compute_quantile_bins(X["total_spent"], q=5)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = add_rfm_features(
            X, r_bins=self.r_bins_, f_bins=self.f_bins_, m_bins=self.m_bins_
        )
        return add_behavioral_features(data)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """EDA ve CSV dışa aktarımı için RFM + davranışsal özellikleri üretir."""
    data = add_rfm_features(df)
    return add_behavioral_features(data)


def load_or_build_features(
    raw_path: Path = RAW_CSV,
    output_path: Path = FEATURES_CSV,
    save: bool = True,
) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_path)
    df_feat = build_features(df_raw)
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(output_path, index=False)
    return df_feat


if __name__ == "__main__":
    df_result = load_or_build_features()
    print(f"[OK] Feature Engineering tamamlandı! Yeni boyut: {df_result.shape}")