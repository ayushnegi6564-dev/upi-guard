"""Builds the simulated UPI transaction table that UPI Guard learns from.

No real transaction data is used. Each column is drawn from a documented
UPI fraud indicator - unusual hour, new payee, new device, an amount far
above the account's normal spend, rapid repeat attempts, failed PINs and
distance from the usual location.
"""
import os

import numpy as np
import pandas as pd

N = 6000
FRAUD_RATE = 0.06
SEED = 2026

# Share of genuine transactions deliberately made to look suspicious
# (fee payments, a first rent transfer, travel). Without this slice the
# problem separates too cleanly to be interesting.
ODD_BUT_GENUINE = 0.05

FEATURES = ["amount", "amount_ratio", "hour", "new_payee", "new_device",
            "txn_last_hour", "failed_pin_attempts", "distance_km",
            "account_age_days"]


def hour_probs(night_w, day_w, eve_w):
    """Hour-of-day distribution: hours 0-4 night, 5-18 day, 19-23 evening."""
    p = np.r_[[night_w] * 5, [day_w] * 14, [eve_w] * 5]
    return p / p.sum()


def build(n=N, fraud_rate=FRAUD_RATE, seed=SEED):
    rng = np.random.default_rng(seed)
    is_fraud = rng.binomial(1, fraud_rate, n)

    # the account's normal spend, and the amount of this transaction
    usual = rng.lognormal(6.2, 0.5, n)                    # ~ Rs 300 - 1500
    amount = np.where(is_fraud == 1,
                      usual * rng.uniform(3, 14, n),
                      usual * rng.uniform(0.2, 2.2, n))
    hour = np.where(is_fraud == 1,
                    rng.choice(range(24), n, p=hour_probs(0.070, 0.018, 0.045)),
                    rng.choice(range(24), n, p=hour_probs(0.008, 0.055, 0.038)))
    new_payee = rng.binomial(1, np.where(is_fraud == 1, 0.88, 0.22))
    new_device = rng.binomial(1, np.where(is_fraud == 1, 0.71, 0.06))
    txn_last_hour = np.where(is_fraud == 1, rng.poisson(4.1, n), rng.poisson(0.7, n))
    failed_pin = np.where(is_fraud == 1, rng.poisson(1.3, n), rng.poisson(0.12, n))
    distance_km = np.where(is_fraud == 1, rng.exponential(340, n), rng.exponential(18, n))
    account_age_days = rng.integers(20, 2800, n)

    df = pd.DataFrame(dict(amount=amount.round(0),
                           amount_ratio=(amount / usual).round(2),
                           hour=hour, new_payee=new_payee, new_device=new_device,
                           txn_last_hour=txn_last_hour,
                           failed_pin_attempts=failed_pin,
                           distance_km=distance_km.round(1),
                           account_age_days=account_age_days,
                           is_fraud=is_fraud))

    # genuine transactions that tick several fraud boxes
    mask = (df.is_fraud == 0) & (rng.random(n) < ODD_BUT_GENUINE)
    k = int(mask.sum())
    df.loc[mask, "amount_ratio"] = rng.uniform(4, 10, k).round(2)
    df.loc[mask, "new_payee"] = 1
    df.loc[mask, "distance_km"] = rng.uniform(150, 700, k).round(1)
    df.loc[mask, "amount"] = (df.loc[mask, "amount_ratio"] * usual[mask]).round(0)
    return df


if __name__ == "__main__":
    df = build()
    here = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(here, "transactions.csv"), index=False)
    print(df.head(), "\n")
    print("transactions: %d  fraud: %d (%.1f%%)"
          % (len(df), df.is_fraud.sum(), 100 * df.is_fraud.mean()))
