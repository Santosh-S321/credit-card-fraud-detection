from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(hours=24)


@dataclass
class CardState:
    n: int = 0                                    # past transactions on this card
    total: float = 0.0                            # their total amount
    last_time: Optional[pd.Timestamp] = None      # most recent transaction time
    recent: deque = field(default_factory=deque)  # (time, amount) from roughly the last 24h


class CardHistoryStore:
    """Keeps just enough per-card history to compute features for a new transaction.

    Must produce exactly the same values as src.features.add_features,
    which tests/test_serving.py checks.
    """

    def __init__(self):
        self._cards: dict[int, CardState] = {}

    def seed(self, df: pd.DataFrame) -> None:
        """Load history from past transactions (e.g. the training CSV)."""
        df = df.sort_values(["cc_num", "trans_date_trans_time"])
        summary = df.groupby("cc_num").agg(n=("amt", "size"), total=("amt", "sum"),
                                           last_time=("trans_date_trans_time", "max"))
        last = df["cc_num"].map(summary["last_time"])
        recent = df[df["trans_date_trans_time"] >= last - DAY]
        recent_by_card = {cc: deque(zip(g["trans_date_trans_time"], g["amt"]))
                          for cc, g in recent.groupby("cc_num")}

        for cc, row in summary.iterrows():
            self._cards[cc] = CardState(n=int(row["n"]), total=float(row["total"]),
                                        last_time=row["last_time"],
                                        recent=recent_by_card.get(cc, deque()))

    def features(self, cc_num: int, time: pd.Timestamp, amt: float) -> dict:
        """Features for a new transaction, using only this card's earlier history."""
        state = self._cards.get(cc_num, CardState())
        if state.last_time is not None and time < state.last_time:
            raise ValueError(f"Transaction at {time} is earlier than this card's "
                             f"last transaction at {state.last_time}")

        # Same windows as rolling(closed="left"): from (time - window) up to, but not including, time
        last_24h = [(t, a) for t, a in state.recent if time - DAY <= t < time]
        last_1h = [(t, a) for t, a in last_24h if t >= time - HOUR]

        return {
            "amt": amt,
            "hour": time.hour,
            "hours_since_prev": ((time - state.last_time).total_seconds() / 3600
                                 if state.n else np.nan),
            "amt_vs_card_avg": amt / (state.total / state.n) if state.n else np.nan,
            "txn_count_1h": float(len(last_1h)),
            "amt_sum_1h": float(sum(a for _, a in last_1h)),
            "txn_count_24h": float(len(last_24h)),
            "amt_sum_24h": float(sum(a for _, a in last_24h)),
        }

    def update(self, cc_num: int, time: pd.Timestamp, amt: float) -> None:
        """Record a transaction so it becomes history for the next one."""
        state = self._cards.setdefault(cc_num, CardState())
        state.n += 1
        state.total += amt
        state.last_time = time
        state.recent.append((time, amt))
        while state.recent and state.recent[0][0] < time - DAY:
            state.recent.popleft()