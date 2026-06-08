import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from feature_preprocessing import preprocess_all

# ── 1. load & preprocess ──────────────────────────────────────────────────────
datasets, le = preprocess_all(
    temporal_path   = "temporal_features.csv",
    http_path       = "http_features.csv",
    tls_path        = "tls_features.csv",
    behavioral_path = "behavioral_features.csv",
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees":   ExtraTreesClassifier(n_estimators=200, random_state=42),
    "XGBoost":      XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss"),
    "LightGBM":     LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),

}

for model_name, clf in models.items():
    for feat_name, ds in datasets.items():
        clf.fit(ds["X_train"], ds["y_train"])
        preds = clf.predict(ds["X_test"])
        print(f"\n{model_name} | {feat_name}")
        print(classification_report(ds["y_test"], preds, target_names=le.classes_))