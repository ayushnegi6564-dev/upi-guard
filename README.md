# UPI Guard

A machine-learning tool that scores a UPI transaction and estimates how
likely it is to be fraudulent. Built as the FAI mini project (Practical 10)
by **Ayush Negi** — B.Tech Cyber Security, Karnavati University.

Live portfolio: <https://sites.google.com/view/upi-guard-ai-mini-project>

## What it does

Nine signals go in — amount, amount against the account's normal spend,
hour of day, new payee, new device, transactions in the last hour, failed
PIN attempts, distance from the usual location and account age — and a
random forest returns the probability that the transaction is fraud.

```
$ python predict.py
FRAUD   99.7%  late-night transfer to a new payee on a new phone
ok       0.0%  the usual evening payment to a saved payee
```

## Results

Trained on 6,000 transactions (6.4% fraudulent), tested on the 1,500 it
had never seen:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9827 | 0.7917 | 0.9896 | 0.8796 | 0.9987 | 0.9830 |
| **Random Forest** | **0.9913** | **0.9368** | **0.9271** | **0.9319** | **0.9995** | **0.9931** |

Average precision is the number to read here, not accuracy: with fraud at
6.4%, a model that flags nothing at all still scores 93.6% accuracy.

The forest leans hardest on `amount_ratio`, `amount`, `txn_last_hour`,
`distance_km` and `new_payee`. The strongest single signal is the amount
*relative to what that account normally spends*, not the amount by itself.

## The threshold is the real decision

`train.py` measures what moving the cut-off costs, on the same 1,500
transactions:

| Threshold | Recall | Precision | Fraud missed | Genuine blocked |
|---|---|---|---|---|
| 0.3 | 0.958 | 0.920 | 4 | 8 |
| 0.5 | 0.927 | 0.937 | 7 | 6 |
| 0.7 | 0.896 | 1.000 | 10 | 0 |
| 0.9 | 0.760 | 1.000 | 23 | 0 |

At 0.7 the model blocks no genuine payment at all but lets three more
frauds through than at 0.5. Which column matters more is a bank's policy
question, not a modelling one.

## Honest limitations

- **The data is simulated, not real.** `simulate.py` draws each column
  from documented UPI fraud indicators; no real transaction data is used,
  and none was reachable from the build environment. Real fraud adapts to
  whatever the bank is checking, which a fixed generator cannot imitate.
- About 5% of the genuine transactions are deliberately made to look
  suspicious — a first rent transfer, a fee payment, travel. Those are
  where most of the six false alarms come from, and they are the honest
  part of the result.
- Nine features only. A production system would also weigh the device
  fingerprint, the payee's own history and the account's full timeline.
- The model scores one transaction at a time and has no memory of the
  account's recent behaviour beyond the counts handed to it.

## Running it

```bash
pip install -r requirements.txt
python simulate.py     # writes transactions.csv
python train.py        # trains both models, writes figures/ and metrics.json
python predict.py      # scores the built-in examples
python app.py          # web interface on http://127.0.0.1:8083
```

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Builds the simulated transaction table, odd-but-genuine slice included |
| `train.py` | Trains and compares both models, writes `model.joblib`, `metrics.json`, `figures/` |
| `predict.py` | Command-line scoring |
| `app.py` | Flask web interface |
| `metrics.json` | The scores and the threshold study, as written by the last run |
