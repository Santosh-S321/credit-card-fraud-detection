"""Fraud scoring API with a demo page.

Run from the project root:
    uvicorn api.app:app --reload
Then open http://127.0.0.1:8000 (demo) or http://127.0.0.1:8000/docs (API docs)
"""
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.explain import describe_reasons, shap_values
from src.features import FEATURES
from src.policy import ACTIONS, expected_costs, realized_cost
from src.serving import CardHistoryStore

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
HISTORY_CSV = os.getenv("HISTORY_CSV", "data/raw/fraudTrain.csv")
DEMO_CSV = Path(os.getenv("DEMO_CSV", "data/raw/fraudTest.csv"))
STATIC_DIR = Path(__file__).parent / "static"
MAX_SCAN = 5000  # most transactions one "until next fraud" replay may process

state = {}                    # filled once at startup
lock = threading.Lock()       # one request at a time may read and update card history
demo_lock = threading.Lock()  # one replay at a time may move the demo cursor


def new_totals():
    return {"replayed": 0, "frauds": 0, "frauds_intercepted": 0, "genuine_otp": 0,
            "genuine_blocked": 0, "fraud_value": 0.0, "cost": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    state["metadata"] = metadata
    state["policy"] = json.loads((MODEL_DIR / "policy.json").read_text())
    state["model"] = joblib.load(MODEL_DIR / metadata["model_file"])

    history = pd.read_csv(HISTORY_CSV, usecols=["cc_num", "trans_date_trans_time", "amt"],
                          parse_dates=["trans_date_trans_time"])
    store = CardHistoryStore()
    store.seed(history)
    state["store"] = store
    state["cards_seeded"] = int(history["cc_num"].nunique())

    if DEMO_CSV.exists():
        demo = pd.read_csv(DEMO_CSV, usecols=["cc_num", "trans_date_trans_time", "amt",
                                              "category", "is_fraud"],
                           parse_dates=["trans_date_trans_time"])
        demo = demo.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
        state["demo"] = {"rows": demo, "cursor": 0, "totals": new_totals()}
    yield
    state.clear()


app = FastAPI(title="Card Fraud Scoring API", version="1.1", lifespan=lifespan)


class Transaction(BaseModel):
    cc_num: int = Field(..., description="Card number, used to look up the card's history")
    trans_date_trans_time: datetime = Field(..., description="Local time, no timezone")
    amt: float = Field(..., gt=0)
    category: str


class Reason(BaseModel):
    feature: str
    text: str
    contribution: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    action: str
    expected_costs: Dict[str, float]
    guardrail_applied: bool
    recorded: bool
    reasons: List[Reason]
    features: Dict[str, Optional[Union[int, float, str]]]


def score_transaction(tx: Transaction, record: bool) -> ScoreResponse:
    """The full decision for one transaction. Used by /score and the demo replay."""
    meta, policy, model = state["metadata"], state["policy"], state["model"]
    if tx.category not in meta["categories"]:
        raise HTTPException(422, f"Unknown category '{tx.category}'. Valid: {meta['categories']}")

    time = pd.Timestamp(tx.trans_date_trans_time.replace(tzinfo=None))
    with lock:
        try:
            feats = state["store"].features(tx.cc_num, time, tx.amt)
        except ValueError as err:
            raise HTTPException(409, str(err))
        if record:
            state["store"].update(tx.cc_num, time, tx.amt)

    row = {**feats, "category": tx.category}
    X = pd.DataFrame([row])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])

    costs = expected_costs([p], [tx.amt], policy["costs"])[0]
    action = str(ACTIONS[costs.argmin()])

    # Trees have no data above the largest training amount, so never auto-approve there
    guardrail = tx.amt > meta["max_amount_seen"] and action == "approve"
    if guardrail:
        action = "otp"

    shap, _ = shap_values(model, X)
    return ScoreResponse(
        fraud_probability=round(p, 6),
        action=action,
        expected_costs={a: round(float(c), 4) for a, c in zip(ACTIONS, costs)},
        guardrail_applied=guardrail,
        recorded=record,
        reasons=describe_reasons(shap.iloc[0], X.iloc[0]),
        features={k: (None if pd.isna(v) else v) for k, v in row.items()},
    )


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    meta = state["metadata"]
    return {"status": "ok", "model": meta["model_file"], "trained_at": meta["trained_at"],
            "history_through": meta["data_to"], "cards_seeded": state["cards_seeded"],
            "decision_rule": state["policy"].get("decision_rule", "expected_cost"),
            "categories": meta["categories"], "demo_available": "demo" in state}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction,
          record: bool = Query(True, description="Add this transaction to the card's history")):
    return score_transaction(tx, record)


def demo_or_404():
    if "demo" not in state:
        raise HTTPException(404, f"Demo data not found at {DEMO_CSV}")
    return state["demo"]


@app.get("/demo/status")
def demo_status():
    demo = demo_or_404()
    return {"cursor": demo["cursor"], "total_rows": len(demo["rows"]), "totals": demo["totals"]}


@app.post("/demo/replay")
def demo_replay(n: int = Query(25, ge=1, le=200), until_fraud: bool = False):
    """Score the next test transactions in time order, as if they were live traffic.

    Fraud labels are used only to grade decisions, never to make them.
    """
    demo = demo_or_404()
    rows, totals, costs = demo["rows"], demo["totals"], state["policy"]["costs"]
    results, processed, skipped = [], 0, 0

    with demo_lock:
        for _ in range(MAX_SCAN if until_fraud else n):
            if demo["cursor"] >= len(rows):
                break
            r = rows.iloc[demo["cursor"]]
            demo["cursor"] += 1
            tx = Transaction(cc_num=int(r["cc_num"]), amt=float(r["amt"]), category=r["category"],
                             trans_date_trans_time=r["trans_date_trans_time"].to_pydatetime())
            try:
                result = score_transaction(tx, record=True)
            except HTTPException as err:
                if err.status_code == 409:  # card already has later history (e.g. from manual tests)
                    skipped += 1
                    continue
                raise
            processed += 1
            is_fraud = bool(r["is_fraud"])

            loss, friction = realized_cost(np.array([result.action]), np.array([int(is_fraud)]),
                                           np.array([tx.amt]), costs)
            totals["replayed"] += 1
            totals["cost"] += loss + friction
            if is_fraud:
                totals["frauds"] += 1
                totals["fraud_value"] += tx.amt
                totals["frauds_intercepted"] += int(result.action != "approve")
            else:
                totals["genuine_otp"] += int(result.action == "otp")
                totals["genuine_blocked"] += int(result.action == "block")

            interesting = is_fraud or result.action != "approve"
            if not until_fraud or interesting:
                results.append({"time": tx.trans_date_trans_time.isoformat(sep=" "),
                                "card": str(tx.cc_num)[-4:], "category": tx.category,
                                "amt": tx.amt, "is_fraud": is_fraud, "result": result})
            if until_fraud and is_fraud:
                break

    return {"processed": processed, "skipped": skipped, "cursor": demo["cursor"],
            "total_rows": len(rows), "rows": results, "totals": totals}