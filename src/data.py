import pandas as pd

from src.features import add_features


def load_train_val(path, cutoff="2020-01-01"):
    """Load the raw training file, add features, and split by time."""
    df = pd.read_csv(path, index_col=0, parse_dates=["trans_date_trans_time"])
    df = add_features(df)
    cutoff = pd.Timestamp(cutoff)
    train = df[df["trans_date_trans_time"] < cutoff].copy()
    val = df[df["trans_date_trans_time"] >= cutoff].copy()
    return train, val