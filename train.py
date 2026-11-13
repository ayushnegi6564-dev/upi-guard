"""Trains UPI Guard, compares two models and saves the better one.

    python train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def make_models():
    """Both models see the same nine features. Class weights matter here
    because only about 6% of the transactions are fraudulent."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, random_state=42,
            class_weight="balanced_subsample"),
    }


def main():
    df = simulate.build()
    df.to_csv(os.path.join(HERE, "transactions.csv"), index=False)
    print("transactions: %d  fraud: %d (%.1f%%)"
          % (len(df), df.is_fraud.sum(), 100 * df.is_fraud.mean()))

    feats = simulate.FEATURES
    X = df[feats].values
    y = df.is_fraud.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)

    models = make_models()
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        results[name] = dict(
            accuracy=round(accuracy_score(yte, pred), 4),
            precision=round(precision_score(yte, pred), 4),
            recall=round(recall_score(yte, pred), 4),
            f1=round(f1_score(yte, pred), 4),
            auc=round(auc(fpr, tpr), 4),
            avg_precision=round(average_precision_score(yte, proba), 4),
            confusion=confusion_matrix(yte, pred).tolist())
        results[name]["_roc"] = (fpr.tolist(), tpr.tolist())
        results[name]["_proba"] = proba.tolist()
        print("\n== %s ==" % name)
        print(classification_report(yte, pred, target_names=["genuine", "fraud"],
                                    digits=4))

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = models[best_name]
    print("best model:", best_name)

    # --- ROC curve -------------------------------------------------------
    plt.figure(figsize=(5.2, 4))
    for name in results:
        fpr, tpr = results[name]["_roc"]
        plt.plot(fpr, tpr, lw=2, label="%s (AUC %.3f)" % (name, results[name]["auc"]))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - UPI Guard")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc.png"), dpi=160)
    plt.close()

    # --- precision-recall: the honest view when fraud is rare ------------
    proba = np.array(results[best_name]["_proba"])
    prec, rec, _ = precision_recall_curve(yte, proba)
    plt.figure(figsize=(5.2, 4))
    plt.plot(rec, prec, lw=2, color="#1f7a4d")
    plt.xlabel("Recall (fraud caught)")
    plt.ylabel("Precision (alerts that were real)")
    plt.title("Precision-recall trade-off - %s" % best_name)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pr.png"), dpi=160)
    plt.close()

    # --- confusion matrix ------------------------------------------------
    cm = np.array(results[best_name]["confusion"])
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["genuine", "fraud"])
    plt.yticks([0, 1], ["genuine", "fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix - " + best_name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion.png"), dpi=160)
    plt.close()

    # --- which signals the forest actually leans on ----------------------
    rf = models["Random Forest"]
    imp = rf.feature_importances_
    order = np.argsort(imp)
    plt.figure(figsize=(5.6, 4))
    plt.barh([feats[i] for i in order], imp[order], color="#1f7a4d")
    plt.xlabel("Importance")
    plt.title("Strongest fraud signals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "importance.png"), dpi=160)
    plt.close()

    # --- the dial a bank would actually tune -----------------------------
    print("\nthreshold study")
    table = []
    for thr in THRESHOLDS:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        table.append(dict(threshold=thr,
                          recall=round(recall_score(yte, pred), 4),
                          precision=round(precision_score(yte, pred, zero_division=0), 4),
                          fraud_missed=int(fn), genuine_blocked=int(fp)))
        print("  thr %.1f  recall %.3f  precision %.3f  missed %3d  wrongly blocked %3d"
              % (thr, table[-1]["recall"], table[-1]["precision"], fn, fp))

    joblib.dump(best, os.path.join(HERE, "model.joblib"))
    for r in results.values():
        r.pop("_roc", None)
        r.pop("_proba", None)
    json.dump({"best": best_name, "results": results, "threshold_study": table,
               "fraud_rate": round(float(df.is_fraud.mean()), 4),
               "top_features": [feats[i] for i in np.argsort(imp)[::-1][:5]]},
              open(os.path.join(HERE, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
