"""
Cross-Validation, Model Metrik Karsilastirmasi ve MLflow Deney Takibi.

Kullanim (Komut satirindan):
    python -m src.train

MLflow Arayuzunu Acmak Icin:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import os
import warnings

# MLflow yeni surum dosya sistemi kisitlamasini devre disi birak
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.config import DATA_DIR, RAW_CSV, ROOT
from src.feature_engineering import build_features, FEATURES_CSV

SEED = 42
MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_churn_pipeline.joblib"
METRICS_CSV_PATH = DATA_DIR / "processed" / "cv_model_comparison.csv"

# Veri sizintisini (Data Leakage) onlemek icin modele girmeyecek sutunlar
LEAKAGE_AND_ID_COLS = {
    "customer_id",
    "snapshot_date",
    "first_purchase_date",
    "last_purchase_date",
    "churn",
    "loyalty_score",
    "loyalty_segment",
    "expected_season_orders",
    "top_season_product",
    "expected_winter_orders",
    "top_winter_product",
}


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Modele girecek sayisal ve kategorik sutunlari otomatik ayirir."""
    drop_cols = set(LEAKAGE_AND_ID_COLS)
    drop_cols |= {
        c for c in df.columns
        if c.startswith("season_prob_") or c.startswith("winter_prob_")
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Eksik verileri dolduran ve olcekleme/One-Hot donusumu yapan Scikit-Learn onisleyici."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def get_candidate_models(scale_pos_weight: float) -> dict[str, object]:
    """Karsilastirilacak Baseline ve Gelismis Modeller."""
    return {
        "LogisticRegression_Baseline": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "RandomForest_Baseline": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=SEED
        ),
        "XGBoost_Standard": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost_Balanced": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def run_experiments_and_track(n_splits: int = 5) -> pd.DataFrame:
    """
    1. Ham veriyi okur ve RFM + Davranissal ozellikleri uretir.
    2. 5-Fold Stratified Cross-Validation ile tum modelleri yaristirir.
    3. Her modeli ve metriklerini MLflow SQLite veritabanina kaydeder.
    4. En iyi modeli models/best_churn_pipeline.joblib olarak disari aktarir.
    """
    print("[1/4] Veri okunuyor ve RFM + Davranissal ozellikler uretiliyor...")
    df_raw = pd.read_csv(RAW_CSV)
    df = build_features(df_raw)

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)

    num_cols, cat_cols = get_feature_lists(df)
    feature_cols = num_cols + cat_cols

    X = df[feature_cols]
    y = df["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    neg_count = int((y_train == 0).sum())
    pos_count = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = round(float(neg_count / pos_count), 2)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # MLflow yeni surum standart SQLite veritabani baglantisi
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Tekstil_Churn_RFM_Model_Karsilastirmasi")

    models = get_candidate_models(scale_pos_weight=scale_pos_weight)
    results = []
    best_score = -1.0
    best_model_name = ""
    best_fitted_pipeline = None

    print(
        f"[2/4] {len(models)} model icin {n_splits}-Fold Cross-Validation "
        f"ve MLflow kaydi basliyor...\n"
    )

    for model_name, clf in models.items():
        with mlflow.start_run(run_name=model_name):
            preprocessor = build_preprocessor(num_cols, cat_cols)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ])

            # 5-Katli Capraz Dogrulama (Cross-Validation)
            cv_res = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )

            cv_metrics = {
                "cv_accuracy": round(float(np.mean(cv_res["test_accuracy"])), 4),
                "cv_precision": round(float(np.mean(cv_res["test_precision"])), 4),
                "cv_recall": round(float(np.mean(cv_res["test_recall"])), 4),
                "cv_f1": round(float(np.mean(cv_res["test_f1"])), 4),
                "cv_roc_auc": round(float(np.mean(cv_res["test_roc_auc"])), 4),
                "cv_f1_std": round(float(np.std(cv_res["test_f1"])), 4),
            }

            # Egitim setiyle modeli egit ve %20 ayrilmis test setinde degerlendir
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            test_metrics = {
                "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "test_precision": round(
                    float(precision_score(y_test, y_pred, zero_division=0)), 4
                ),
                "test_recall": round(
                    float(recall_score(y_test, y_pred, zero_division=0)), 4
                ),
                "test_f1": round(
                    float(f1_score(y_test, y_pred, zero_division=0)), 4
                ),
                "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            }

            # MLflow parametre ve metrik kayitlari
            mlflow.log_params({
                "model_name": model_name,
                "cv_folds": n_splits,
                "num_features": len(num_cols),
                "cat_features": len(cat_cols),
                "total_features": len(feature_cols),
                "scale_pos_weight": scale_pos_weight,
            })
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics(test_metrics)

            row = {"model": model_name, **cv_metrics, **test_metrics}
            results.append(row)

            print(
                f" -> {model_name:28s} | CV F1: {cv_metrics['cv_f1']:.4f} | "
                f"CV Recall: {cv_metrics['cv_recall']:.4f} | "
                f"CV ROC-AUC: {cv_metrics['cv_roc_auc']:.4f} | "
                f"Test F1: {test_metrics['test_f1']:.4f}"
            )

            # En iyi modeli CV F1 (%60) ve CV ROC-AUC (%40) agirlikli puana gore sec
            composite_score = (
                0.60 * cv_metrics["cv_f1"] + 0.40 * cv_metrics["cv_roc_auc"]
            )
            if composite_score > best_score:
                best_score = composite_score
                best_model_name = model_name
                best_fitted_pipeline = pipeline

    results_df = pd.DataFrame(results).sort_values(by="cv_f1", ascending=False)
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_CSV_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_fitted_pipeline, BEST_MODEL_PATH)

    print(f"\n[3/4] Karsilastirma tablosu kaydedildi: {METRICS_CSV_PATH}")
    print(f"[4/4] En basarili model ({best_model_name}) kaydedildi: {BEST_MODEL_PATH}\n")
    print("=== TUM MODELLERIN KARSILASTIRMA TABLOSU ===")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_experiments_and_track()