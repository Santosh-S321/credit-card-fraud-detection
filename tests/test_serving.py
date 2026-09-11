import numpy as np
import pandas as pd
import pytest

from src.features import NUMERIC, add_features
from src.serving import CardHistoryStore

# Gaps in minutes chosen to land exactly on window edges (0, 60, 1440) and just around them
GAPS = [0, 1, 30, 59, 60, 61, 600, 1439, 1440, 1441, 3000]


def make_transactions(cards=6, per_card=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for card in range(cards):
        t = pd.Timestamp("2020-01-01")
        for _ in range(per_card):
            t += pd.Timedelta(minutes=int(rng.choice(GAPS)))
            rows.append({"cc_num": 1000 + card, "trans_date_trans_time": t,
                         "amt": round(float(rng.exponential(80)) + 1, 2), "category": "misc_net",
                         "lat": 0.0, "long": 0.0, "merch_lat": 0.0, "merch_long": 0.0})
    return pd.DataFrame(rows)


def replay(store, df):
    out = []
    for row in df.itertuples():
        out.append(store.features(row.cc_num, row.trans_date_trans_time, row.amt))
        store.update(row.cc_num, row.trans_date_trans_time, row.amt)
    return pd.DataFrame(out)


def assert_same(expected: pd.DataFrame, actual: pd.DataFrame):
    for col in NUMERIC:
        assert np.allclose(expected[col], actual[col], equal_nan=True), f"mismatch in {col}"


def test_store_matches_training_features():
    df = make_transactions()
    expected = add_features(df)                      # sorted by card, then time
    actual = replay(CardHistoryStore(), expected)    # replay in that same order
    assert_same(expected, actual)


def test_seeded_store_matches_training_features():
    df = add_features(make_transactions())
    cutoff = df["trans_date_trans_time"].quantile(0.6)
    history, new = df[df["trans_date_trans_time"] < cutoff], df[df["trans_date_trans_time"] >= cutoff]

    store = CardHistoryStore()
    store.seed(history)
    assert_same(new.reset_index(drop=True), replay(store, new))


def test_rejects_out_of_order_transactions():
    store = CardHistoryStore()
    store.update(1, pd.Timestamp("2020-01-02"), 10.0)
    with pytest.raises(ValueError):
        store.features(1, pd.Timestamp("2020-01-01"), 10.0)