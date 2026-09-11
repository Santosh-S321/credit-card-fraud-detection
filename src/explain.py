import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import FEATURES


def shap_values(model, X: pd.DataFrame):
    """Exact TreeSHAP contributions (in log-odds) for each transaction.

    Returns a DataFrame with one column per original feature, plus the
    base value: the model's average output before seeing any features.
    """
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(X[FEATURES])
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    contribs = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
    shap = pd.DataFrame(contribs[:, :-1], columns=names, index=X.index)
    base = float(contribs[0, -1])

    # SHAP values add up, so the one-hot columns merge into one "category" value
    cat_cols = [c for c in names if c.startswith("category_")]
    shap["category"] = shap[cat_cols].sum(axis=1)
    return shap.drop(columns=cat_cols), base


def top_reasons(shap_row: pd.Series, feature_row: pd.Series, k=3):
    """The k features pushing this transaction most strongly toward fraud."""
    pushes = shap_row[shap_row > 0].sort_values(ascending=False).head(k)
    reasons = []
    for f, v in pushes.items():
        value = feature_row[f]
        value = value.item() if hasattr(value, "item") else value  # NumPy → plain Python
        reasons.append((f, value, round(float(v), 2)))
    return reasons


FIRST_TRANSACTION = "First transaction on this card"

REASON_TEXT = {
    "amt": lambda v: f"Amount ${v:,.2f}",
    "amt_vs_card_avg": lambda v: f"{v:.1f}x this card's average spend",
    "hours_since_prev": lambda v: (f"{v * 60:.0f} minutes since this card's previous transaction"
                                   if v < 1 else f"{v:.1f} hours since this card's previous transaction"),
    "amt_sum_24h": lambda v: (f"${v:,.2f} spent on this card in the previous 24 hours"
                              if v > 0 else "No other transactions in the previous 24 hours"),
    "txn_count_24h": lambda v: (f"{int(v)} transactions on this card in the previous 24 hours"
                                if v > 0 else "No other transactions in the previous 24 hours"),
    "amt_sum_1h": lambda v: (f"${v:,.2f} spent on this card in the previous hour"
                             if v > 0 else "No other transactions in the previous hour"),
    "txn_count_1h": lambda v: (f"{int(v)} transactions on this card in the previous hour"
                               if v > 0 else "No other transactions in the previous hour"),
    "hour": lambda v: f"Made during the {int(v):02d}:00 hour",
    "category": lambda v: f"Merchant category: {v}",
}


def describe_reasons(shap_row: pd.Series, feature_row: pd.Series, k=3, min_push=0.05):
    """Top k reasons pushing toward fraud, as readable sentences.

    Pushes smaller than min_push (in log-odds) are too weak to count as reasons.
    """
    reasons, seen = [], set()
    for f, v in shap_row[shap_row > min_push].sort_values(ascending=False).items():
        value = feature_row[f]
        if f in ("amt_vs_card_avg", "hours_since_prev") and pd.isna(value):
            text = FIRST_TRANSACTION
        else:
            text = REASON_TEXT[f](value)
        if text in seen:            # two features can describe the same fact
            continue
        seen.add(text)
        reasons.append({"feature": f, "text": text, "contribution": round(float(v), 3)})
        if len(reasons) == k:
            break
    return reasons