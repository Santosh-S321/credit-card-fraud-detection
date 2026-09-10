import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, distance, and per-card history features.

    Every card-history feature uses only transactions strictly
    BEFORE the current one, so nothing from the future leaks in.
    """
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    t = df["trans_date_trans_time"]
    card = df.groupby("cc_num")

    # Context features
    df["hour"] = t.dt.hour
    df["distance_km"] = haversine_km(df["lat"], df["long"],
                                     df["merch_lat"], df["merch_long"])

    # Time since this card's previous transaction
    df["hours_since_prev"] = card["trans_date_trans_time"].diff().dt.total_seconds() / 3600

    # Amount vs this card's average so far (past transactions only)
    past_sum = card["amt"].cumsum() - df["amt"]
    past_n = card.cumcount()
    df["amt_vs_card_avg"] = df["amt"] / (past_sum / past_n.replace(0, np.nan))

    # Velocity: count and total spend in the previous 1h and 24h
    for window in ["1h", "24h"]:
        rolled = (card.rolling(window, on="trans_date_trans_time", closed="left")["amt"]
                      .agg(["count", "sum"]))
        # Result comes back in the same order as df (sorted by card, then time)
        assert (rolled.index.get_level_values("cc_num") == df["cc_num"]).all()
        df[f"txn_count_{window}"] = rolled["count"].fillna(0).to_numpy()
        df[f"amt_sum_{window}"] = rolled["sum"].fillna(0).to_numpy()

    return df

# The model's input contract: the API must send exactly these columns
NUMERIC = ["amt", "hour", "hours_since_prev", "amt_vs_card_avg",
           "txn_count_1h", "amt_sum_1h", "txn_count_24h", "amt_sum_24h"]
CATEGORICAL = ["category"]
FEATURES = NUMERIC + CATEGORICAL