"""Scores a single transaction from the command line.

    python predict.py --amount 48000 --ratio 9.5 --hour 3 --new-payee 1 \
                      --new-device 1 --recent 5 --failed-pin 2 --distance 410 \
                      --account-age 60
    python predict.py            # scores two built-in examples
"""
import argparse
import os
import sys

import joblib
import numpy as np

import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.joblib")

EXAMPLES = [
    ("late-night transfer to a new payee on a new phone",
     dict(amount=48000, amount_ratio=9.5, hour=3, new_payee=1, new_device=1,
          txn_last_hour=5, failed_pin_attempts=2, distance_km=410,
          account_age_days=60)),
    ("the usual evening payment to a saved payee",
     dict(amount=850, amount_ratio=1.1, hour=19, new_payee=0, new_device=0,
          txn_last_hour=1, failed_pin_attempts=0, distance_km=6,
          account_age_days=1400)),
]


def load():
    if not os.path.exists(MODEL):
        sys.exit("model.joblib not found - run 'python train.py' first")
    return joblib.load(MODEL)


def score(model, row):
    x = np.array([[float(row[f]) for f in simulate.FEATURES]])
    return round(float(model.predict_proba(x)[0][1]) * 100, 1)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Score one UPI transaction.")
    p.add_argument("--amount", type=float)
    p.add_argument("--ratio", type=float, help="amount / the account's normal spend")
    p.add_argument("--hour", type=int)
    p.add_argument("--new-payee", type=int, choices=[0, 1])
    p.add_argument("--new-device", type=int, choices=[0, 1])
    p.add_argument("--recent", type=int, help="transactions in the last hour")
    p.add_argument("--failed-pin", type=int)
    p.add_argument("--distance", type=float, help="km from the usual location")
    p.add_argument("--account-age", type=int, help="account age in days")
    return p.parse_args(argv)


def main():
    model = load()
    if len(sys.argv) == 1:
        for label, row in EXAMPLES:
            risk = score(model, row)
            print("%-6s %5.1f%%  %s" % ("FRAUD" if risk >= 50 else "ok", risk, label))
        return
    a = parse_args(sys.argv[1:])
    row = dict(amount=a.amount, amount_ratio=a.ratio, hour=a.hour,
               new_payee=a.new_payee, new_device=a.new_device,
               txn_last_hour=a.recent, failed_pin_attempts=a.failed_pin,
               distance_km=a.distance, account_age_days=a.account_age)
    missing = [k for k, v in row.items() if v is None]
    if missing:
        sys.exit("missing values for: " + ", ".join(missing))
    risk = score(model, row)
    print("%-6s %5.1f%%" % ("FRAUD" if risk >= 50 else "ok", risk))


if __name__ == "__main__":
    main()
