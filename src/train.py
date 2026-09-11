"""Train the deployment model on all labeled training data.

Repeats the validated procedure: early stopping on the most recent
two months to choose the number of trees, then a refit on everything.

Usage, from the project root:
    python -m src.train --data data/raw/fraudTrain.csv --out models
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features import CATEGORICAL, FEATURES, NUMERIC, add_features

PARAMS = dict(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              eval_metric="aucpr", n_jobs=-1, random_state=42)


def make_prep():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def train(df: pd.DataFrame, stop_months: int = 2):
    t = df["trans_date_trans_time"]
    stop_start = t.max() - pd.DateOffset(months=stop_months)
    fit_part, stop_part = df[t < stop_start], df[t >= stop_start]

    prep = make_prep().fit(fit_part[FEATURES])
    finder = XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **PARAMS)
    finder.fit(prep.transform(fit_part[FEATURES]), fit_part["is_fraud"],
               eval_set=[(prep.transform(stop_part[FEATURES]), stop_part["is_fraud"])],
               verbose=False)
    n_trees = finder.best_iteration + 1

    model = Pipeline([("prep", make_prep()),
                      ("clf", XGBClassifier(n_estimators=n_trees, **PARAMS))])
    model.fit(df[FEATURES], df["is_fraud"])
    return model, n_trees, stop_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/fraudTrain.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.data, index_col=0, parse_dates=["trans_date_trans_time"])
    df = add_features(df)
    model, n_trees, stop_start = train(df)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "xgb_deploy.joblib")

    t = df["trans_date_trans_time"]
    metadata = {
        "model_file": "xgb_deploy.joblib",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_from": str(t.min()), "data_to": str(t.max()),
        "early_stop_from": str(stop_start),
        "rows": int(len(df)), "fraud_rate": float(df["is_fraud"].mean()),
        "n_trees": int(n_trees),
        "features": FEATURES,
        "categories": sorted(df["category"].unique().tolist()),
        "max_amount_seen": float(df["amt"].max()),
        "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__},
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()