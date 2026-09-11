import json
import os
from pathlib import Path

import pytest

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
HISTORY_CSV = Path(os.getenv("HISTORY_CSV", "data/raw/fraudTrain.csv"))
READY = (MODEL_DIR / "xgb_deploy.joblib").exists() and HISTORY_CSV.exists()
pytestmark = pytest.mark.skipif(
    not READY, reason="Needs data/raw/fraudTrain.csv and a trained model: python -m src.train")

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

NEW_CARD = 999_000_000_001


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs startup: loads the model and seeds history
        yield c


def tx(**overrides):
    base = {"cc_num": NEW_CARD, "trans_date_trans_time": "2020-07-01T10:00:00",
            "amt": 42.0, "category": "grocery_pos"}
    return {**base, **overrides}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_response_shape(client):
    r = client.post("/score", params={"record": False}, json=tx())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["action"] in {"approve", "otp", "block"}
    assert body["features"]["hours_since_prev"] is None  # brand-new card: no history


def test_record_false_leaves_history_unchanged(client):
    first = client.post("/score", params={"record": False}, json=tx(cc_num=NEW_CARD + 1)).json()
    second = client.post("/score", params={"record": False}, json=tx(cc_num=NEW_CARD + 1)).json()
    assert first == second


def test_recorded_transactions_become_history(client):
    card = NEW_CARD + 2
    client.post("/score", json=tx(cc_num=card, trans_date_trans_time="2020-07-01T10:00:00"))
    body = client.post("/score", params={"record": False},
                       json=tx(cc_num=card, trans_date_trans_time="2020-07-01T10:30:00")).json()
    assert body["features"]["txn_count_1h"] == 1
    assert body["features"]["hours_since_prev"] == pytest.approx(0.5)


def test_never_approves_amounts_beyond_training_range(client):
    max_amount = json.loads((MODEL_DIR / "metadata.json").read_text())["max_amount_seen"]
    body = client.post("/score", params={"record": False},
                       json=tx(cc_num=NEW_CARD + 3, amt=max_amount * 2)).json()
    assert body["action"] != "approve"


def test_rejects_out_of_order_transaction(client):
    card = NEW_CARD + 4
    client.post("/score", json=tx(cc_num=card, trans_date_trans_time="2020-07-02T10:00:00"))
    r = client.post("/score", json=tx(cc_num=card, trans_date_trans_time="2020-07-01T10:00:00"))
    assert r.status_code == 409


@pytest.mark.parametrize("bad", [{"category": "casino"}, {"amt": -5}])
def test_rejects_invalid_input(client, bad):
    assert client.post("/score", json=tx(**bad)).status_code == 422


def test_home_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_demo_replay_advances_and_counts(client):
    if not client.get("/health").json()["demo_available"]:
        pytest.skip("Demo data not available")
    before = client.get("/demo/status").json()
    body = client.post("/demo/replay", params={"n": 5}).json()
    assert body["cursor"] == before["cursor"] + 5
    assert body["totals"]["replayed"] == before["totals"]["replayed"] + body["processed"]
    assert len(body["rows"]) == body["processed"]