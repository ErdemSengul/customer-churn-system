"""
Veri yükleme, sızıntı sütunlarını ayıklama, train/test bölme ve
scikit-learn Pipeline kurulumu.

Kullanım:
    from src.models.pipeline import load_data, split_data, build_pipeline
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "customers.csv"

# --- Hedef değişkenler ---
TARGET_CHURN = "churn"             # sınıflandırma (0/1), model olasılığı = churn riski %
TARGET_LOYALTY = "loyalty_score"   # regresyon (0-100)

# --- Modele ASLA girmeyecek sütunlar ---
ID_COLS = ["customer_id"]
DATE_COLS = ["first_purchase_date", "last_purchase_date"]    # recency_days zaten var
TARGET_COLS = ["churn", "loyalty_score", "loyalty_segment"]  # hedefler ve türevleri
# Kış çıktıları: ürün öneri modeline ait, churn/sadakat modeline girmez
WINTER_COLS_PREFIX = ("winter_prob_",)
WINTER_COLS = ["expected_winter_orders", "top_winter_product"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame, extra_drop=()):
    """Sızıntı ve kimlik sütunlarını çıkarıp (sayısal, kategorik) listelerini döndürür."""
    drop = set(ID_COLS + DATE_COLS + TARGET_COLS + WINTER_COLS + list(extra_drop))
    drop |= {c for c in df.columns if c.startswith(WINTER_COLS_PREFIX)}
    features = [c for c in df.columns if c not in drop]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def split_data(df: pd.DataFrame, target: str = TARGET_CHURN, test_size: float = 0.2,
               extra_drop=()):
    """Train/test böler. Churn için sınıf oranını korur (stratify)."""
    num_cols, cat_cols = get_feature_columns(df, extra_drop)
    X = df[num_cols + cat_cols]
    y = df[target]
    stratify = y if target == TARGET_CHURN else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=stratify
    )
    return X_tr, X_te, y_tr, y_te, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, scale: bool = True) -> ColumnTransformer:
    """Sayısal: median imputer (+ eksik göstergesi) [+ StandardScaler]
    Kategorik: en sık değer imputer + OneHot (nadir şehirler 'infrequent' grubuna)."""
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:  # LogReg/Linear için gerekli, ağaç modellerinde gerekmez
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=20, sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols),
         ("cat", Pipeline(cat_steps), cat_cols)],
        remainder="drop",
    )


def build_pipeline(model, num_cols, cat_cols, scale: bool = True) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols, scale=scale)),
        ("model", model),
    ])


if __name__ == "__main__":
    # Hızlı duman testi
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    df = load_data()
    X_tr, X_te, y_tr, y_te, num_cols, cat_cols = split_data(df)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Sayısal: {len(num_cols)}, Kategorik: {cat_cols}")
    print(f"Churn oranı (train/test): {y_tr.mean():.3f} / {y_te.mean():.3f}")

    for name, model, scale in [
        ("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced"), True),
        ("RandomForest", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED), False),
    ]:
        pipe = build_pipeline(model, num_cols, cat_cols, scale=scale).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        n_feat = pipe.named_steps["prep"].get_feature_names_out().shape[0]
        print(f"{name:13s} ROC-AUC: {auc:.3f}  (dönüşüm sonrası {n_feat} özellik)")