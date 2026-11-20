"""
Baseline (Dummy, Logistic Regression, Random Forest) ve XGBoost Model Eğitimi,
5-Fold Stratified CV, Optimal Eşik (Threshold) Belirleme ve MLflow Deney Takibi.

Kullanım:
    python -m src.train

MLflow Arayüzü:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import os
import warnings

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    FEATURES_CSV,
    METRICS_CSV_PATH,
    MLFLOW_DB_URI,
    MODELS_DIR,
    RAW_CSV,
    SEED,
)
from src.feature_engineering import build_features
from src.pipeline import TARGET_CHURN, build_pipeline, split_data


def load_training_dataframe() -> tuple[pd.DataFrame, str]:
    """Önce PostgreSQL churn_dataset görünümünü dener, bağlantı yoksa RAW_CSV okur."""
    try:
        from src.data_loader import load_customers

        df = load_customers()
        return df, "PostgreSQL (churn_dataset)"
    except Exception as exc:
        if not RAW_CSV.is_file():
            raise FileNotFoundError(
                f"PostgreSQL bağlantısı kurulamadı ({exc}) ve {RAW_CSV} bulunamadı. "
                "Önce 'python -m src.data_loader generate' çalıştırın."
            ) from exc
        return pd.read_csv(RAW_CSV), f"CSV ({RAW_CSV.name})"


def get_candidate_models(scale_pos_weight: float) -> dict[str, tuple[object, bool]]:
    """
    Aday modeller sözlüğü: {model_adi: (estimator, scale_numeric_features)}
    Lineer modellerde StandardScaler açık (True), ağaç modellerinde kapalı (False).
    """
    return {
        "Dummy_Baseline": (
            DummyClassifier(strategy="most_frequent", random_state=SEED),
            False,
        ),
        "LogisticRegression_Baseline": (
            LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED
            ),
            True,
        ),
        "RandomForest_Baseline": (
            RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost_Standard": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost_Balanced": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            ),
            False,
        ),
    }


def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """F1 skorunu maksimize eden olasılık karar eşiğini bulur."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.50
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return round(float(np.clip(thresholds[best_idx], 0.15, 0.85)), 4)


def run_experiments_and_track(n_splits: int = 5) -> pd.DataFrame:
    print("[1/4] Veri yükleniyor...")
    df_raw, source_name = load_training_dataframe()
    print(f"      Kaynak: {source_name} | Satır: {len(df_raw)} | Churn oranı: %{df_raw[TARGET_CHURN].mean()*100:.2f}")

    # EDA ve analiz dosyası için işlenmiş kopyayı güncel tut
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    build_features(df_raw).to_csv(FEATURES_CSV, index=False)

    X_train, X_test, y_train, y_test, num_cols, cat_cols = split_data(
        df_raw, target=TARGET_CHURN, test_size=0.20
    )

    neg_count = int((y_train == 0).sum())
    pos_count = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = round(float(neg_count / pos_count), 2)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("E_Ticaret_Musteri_Churn_Sistemi")

    candidates = get_candidate_models(scale_pos_weight=scale_pos_weight)
    results = []
    best_score = -1.0
    best_bundle = None

    print(f"[2/4] {len(candidates)} model için {n_splits}-Fold CV ve eşik optimizasyonu başlıyor...\n")

    for model_name, (estimator, scale_flag) in candidates.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = build_pipeline(estimator, num_cols, cat_cols, scale=scale_flag)

            cv_res = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )

            # Out-of-fold tahminlerle sızıntısız karar eşiği belirle
            oof_probs = cross_val_predict(
                pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            opt_threshold = (
                0.50 if model_name == "Dummy_Baseline" else find_optimal_threshold(y_train, oof_probs)
            )

            cv_metrics = {
                "cv_accuracy": round(float(np.mean(cv_res["test_accuracy"])), 4),
                "cv_precision": round(float(np.mean(cv_res["test_precision"])), 4),
                "cv_recall": round(float(np.mean(cv_res["test_recall"])), 4),
                "cv_f1": round(float(np.mean(cv_res["test_f1"])), 4),
                "cv_roc_auc": round(float(np.mean(cv_res["test_roc_auc"])), 4),
                "cv_pr_auc": round(float(np.mean(cv_res["test_average_precision"])), 4),
            }

            # Tüm eğitim setiyle eğit ve %20 test setinde değerlendir
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred_default = (y_prob >= 0.50).astype(int)
            y_pred_opt = (y_prob >= opt_threshold).astype(int)

            test_metrics = {
                "opt_threshold": opt_threshold,
                "test_accuracy": round(float(accuracy_score(y_test, y_pred_opt)), 4),
                "test_precision": round(float(precision_score(y_test, y_pred_opt, zero_division=0)), 4),
                "test_recall": round(float(recall_score(y_test, y_pred_opt, zero_division=0)), 4),
                "test_f1_default50": round(float(f1_score(y_test, y_pred_default, zero_division=0)), 4),
                "test_f1_opt": round(float(f1_score(y_test, y_pred_opt, zero_division=0)), 4),
                "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "test_pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
            }

            mlflow.log_params({
                "model_name": model_name,
                "data_source": source_name,
                "cv_folds": n_splits,
                "raw_num_features": len(num_cols),
                "raw_cat_features": len(cat_cols),
                "scale_pos_weight": scale_pos_weight,
                "opt_threshold": opt_threshold,
            })
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
                serialization_format="pickle",
            )

            row = {"model": model_name, **cv_metrics, **test_metrics}
            results.append(row)

            print(
                f" -> {model_name:28s} | CV ROC-AUC: {cv_metrics['cv_roc_auc']:.4f} | "
                f"CV PR-AUC: {cv_metrics['cv_pr_auc']:.4f} | "
                f"Test F1 (opt@{opt_threshold:.2f}): {test_metrics['test_f1_opt']:.4f} "
                f"(def@0.50: {test_metrics['test_f1_default50']:.4f})"
            )

            # Model seçimi: PR-AUC (%40) + ROC-AUC (%35) + Optimize Test F1 (%25)
            composite_score = (
                0.40 * cv_metrics["cv_pr_auc"]
                + 0.35 * cv_metrics["cv_roc_auc"]
                + 0.25 * test_metrics["test_f1_opt"]
            )
            if composite_score > best_score and model_name != "Dummy_Baseline":
                best_score = composite_score
                best_bundle = {
                    "model_name": model_name,
                    "pipeline": pipeline,
                    "threshold": opt_threshold,
                    "raw_feature_cols": num_cols + cat_cols,
                    "cv_metrics": cv_metrics,
                    "test_metrics": test_metrics,
                }

    results_df = pd.DataFrame(results).sort_values(by="cv_pr_auc", ascending=False)
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_CSV_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_bundle, BEST_MODEL_PATH)

    print(f"\n[3/4] Karşılaştırma tablosu kaydedildi: {METRICS_CSV_PATH}")
    print(f"[4/4] En iyi model ({best_bundle['model_name']} | eşik={best_bundle['threshold']}) kaydedildi: {BEST_MODEL_PATH}\n")
    print("=== TÜM MODELLERİN KARŞILAŞTIRMA TABLOSU ===")
    print(results_df.to_string(index=False))
    return results_df


def predict_churn(df_customers: pd.DataFrame) -> pd.DataFrame:
    """Dashboard / API katmanı için eğitilmiş en iyi modelle churn olasılığı ve risk sınıfı üretir."""
    bundle = joblib.load(BEST_MODEL_PATH)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    probs = pipeline.predict_proba(df_customers)[:, 1]
    preds = (probs >= threshold).astype(int)
    risk_tier = pd.cut(
        probs,
        bins=[-0.01, 0.30, threshold, 0.75, 1.01],
        labels=["Dusuk_Risk", "Izlenmeli", "Yuksek_Risk", "Kritik_Risk"],
    ).astype(str)

    return pd.DataFrame({
        "customer_id": df_customers["customer_id"].values if "customer_id" in df_customers else range(len(df_customers)),
        "churn_probability": np.round(probs, 4),
        "churn_prediction": preds,
        "risk_tier": risk_tier,
        "applied_threshold": threshold,
    })


if __name__ == "__main__":
    run_experiments_and_track()