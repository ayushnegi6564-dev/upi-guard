"""Minimal web interface for UPI Guard."""
import os

import joblib
import numpy as np
from flask import Flask, render_template_string, request

import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "model.joblib"))
app = Flask(__name__)

DEFAULTS = dict(amount="48000", amount_ratio="9.5", hour="3", new_payee="1",
                new_device="1", txn_last_hour="5", failed_pin_attempts="2",
                distance_km="410", account_age_days="60")

LABELS = dict(amount="Amount (Rs)", amount_ratio="Amount / normal spend",
              hour="Hour of day (0-23)", new_payee="New payee (1 = yes)",
              new_device="New device (1 = yes)",
              txn_last_hour="Transactions in the last hour",
              failed_pin_attempts="Failed PIN attempts",
              distance_km="Distance from usual location (km)",
              account_age_days="Account age (days)")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>UPI Guard</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial;background:#06140e;color:#e6f4ec;margin:0;padding:40px}
 .card{max-width:760px;margin:0 auto;background:#0e2419;border-radius:14px;padding:28px 32px;
       box-shadow:0 8px 30px rgba(0,0,0,.45)}
 h1{margin:0 0 4px;font-size:26px} .sub{color:#8fc0a8;margin-bottom:22px;font-size:14px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px 18px}
 label{display:block;font-size:13px;color:#8fc0a8;margin-bottom:4px}
 input{width:100%;padding:9px 11px;border-radius:7px;border:1px solid #275c42;
       background:#081c13;color:#e6f4ec;font-size:14px;box-sizing:border-box}
 button{margin-top:18px;padding:11px 22px;border:0;border-radius:8px;background:#1f7a4d;
       color:#fff;font-size:15px;cursor:pointer}
 .verdict{margin-top:24px;padding:18px;border-radius:10px;font-size:19px;font-weight:600}
 .bad{background:#3a1420;border:1px solid #7d2440;color:#ff9db0}
 .good{background:#10281c;border:1px solid #1f6b43;color:#79e0a8}
 .note{margin-top:16px;font-size:13px;color:#8fc0a8;line-height:1.5}
</style></head><body><div class="card">
<h1>UPI Guard</h1>
<div class="sub">Machine-learning check for suspicious UPI transactions &middot; FAI Mini Project</div>
<form method="post"><div class="grid">
{% for f in feats %}
  <div><label>{{ labels[f] }}</label>
  <input type="text" name="{{ f }}" value="{{ values[f] }}"></div>
{% endfor %}
</div><button type="submit">Check this transaction</button></form>
{% if risk is not none %}
  <div class="verdict {{ 'bad' if risk >= 50 else 'good' }}">
    {{ 'Likely FRAUD' if risk >= 50 else 'Looks genuine' }} &mdash; risk score {{ risk }}%
  </div>
  <div class="note">The score is the forest's probability that the transaction is
  fraudulent. A bank would not block at 50%: raising the cut-off to 70% blocks no
  genuine payments at all but lets three more frauds through, which is the
  trade-off the threshold study in the repository measures.</div>
{% endif %}
</div></body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(DEFAULTS)
    risk = None
    if request.method == "POST":
        for f in simulate.FEATURES:
            values[f] = request.form.get(f, DEFAULTS[f])
        try:
            x = np.array([[float(values[f]) for f in simulate.FEATURES]])
            risk = round(float(model.predict_proba(x)[0][1]) * 100, 1)
        except ValueError:
            risk = None
    return render_template_string(PAGE, feats=simulate.FEATURES, labels=LABELS,
                                  values=values, risk=risk)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8083)
