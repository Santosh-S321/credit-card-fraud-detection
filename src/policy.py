import numpy as np

ACTIONS = np.array(["approve", "otp", "block"])


def expected_costs(p, amt, costs):
    """Expected cost of each action for each transaction, given fraud probability p.

    Columns: approve, otp, block.
    """
    p, amt = np.asarray(p, dtype=float), np.asarray(amt, dtype=float)
    approve = p * amt
    otp = p * (1 - costs["otp_stop_rate"]) * amt + (1 - p) * costs["otp_friction"]
    block = (1 - p) * costs["block_friction"]
    return np.column_stack([approve, otp, block])


def decide_expected_cost(p, amt, costs):
    """Pick the cheapest action per transaction. No thresholds to tune."""
    return ACTIONS[expected_costs(p, amt, costs).argmin(axis=1)]


def decide_thresholds(p, t_otp, t_block):
    """Fixed score thresholds, the same for every amount."""
    p = np.asarray(p, dtype=float)
    return np.select([p >= t_block, p >= t_otp], ["block", "otp"], default="approve")


def realized_cost(actions, y, amt, costs):
    """Actual fraud loss and customer friction for a set of decisions with known labels."""
    y, amt = np.asarray(y), np.asarray(amt, dtype=float)
    fraud, legit = y == 1, y == 0
    loss = (amt[fraud & (actions == "approve")].sum()
            + (1 - costs["otp_stop_rate"]) * amt[fraud & (actions == "otp")].sum())
    friction = (costs["otp_friction"] * (legit & (actions == "otp")).sum()
                + costs["block_friction"] * (legit & (actions == "block")).sum())
    return float(loss), float(friction)