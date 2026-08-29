"""
Credit Risk PD Model — build, validate, and backtest.
Dataset: UCI "Default of Credit Card Clients" (Taiwan, 2005), 30,000 obs.
Target: default.payment.next.month (1 = default next month, 0 = no default)
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              precision_recall_curve, confusion_matrix,
                              classification_report)
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT = "/home/claude/artifacts"
import os
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ----------------------------------------------------------------------------
df = pd.read_csv("/home/claude/UCI_Credit_Card.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"default.payment.next.month": "DEFAULT", "PAY_0": "PAY_1"})
df = df.drop(columns=["ID"])

# Known data-quality issues in this public dataset (documented in UCI notes):
# EDUCATION has undocumented codes 0,5,6 -> collapse into "other" (4)
# MARRIAGE has undocumented code 0 -> collapse into "other" (3)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

data_quality_notes = {
    "n_rows": int(len(df)),
    "n_cols": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_prevalence": float(df["DEFAULT"].mean()),
    "education_recode": "0,5,6 -> 4 (undocumented/other codes per UCI dataset notes)",
    "marriage_recode": "0 -> 3 (undocumented code)",
}

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
pay_cols = [f"PAY_{i}" for i in [1, 2, 3, 4, 5, 6]]
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
payamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

df["MAX_DELINQ"] = df[pay_cols].max(axis=1)                      # worst delinquency status, 6mo
df["N_MONTHS_DELINQ"] = (df[pay_cols] > 0).sum(axis=1)            # months >=1 late
df["AVG_BILL"] = df[bill_cols].mean(axis=1)
df["AVG_PAY_AMT"] = df[payamt_cols].mean(axis=1)
df["UTILIZATION"] = (df["AVG_BILL"] / df["LIMIT_BAL"].replace(0, np.nan)).clip(-2, 5).fillna(0)
df["PAY_TO_BILL_RATIO"] = (df["AVG_PAY_AMT"] / df["AVG_BILL"].replace(0, np.nan)).clip(-2, 5).fillna(0)
df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]              # recent vs 6mo-ago balance

feature_cols = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *pay_cols, *bill_cols, *payamt_cols,
    "MAX_DELINQ", "N_MONTHS_DELINQ", "AVG_BILL", "AVG_PAY_AMT",
    "UTILIZATION", "PAY_TO_BILL_RATIO", "BILL_TREND",
]
X = df[feature_cols].copy()
y = df["DEFAULT"].copy()

# ----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (development sample vs out-of-sample holdout)
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------------------------------------------------------
# 4. MODEL 1 — Logistic Regression (industry-standard scorecard baseline)
# ----------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
logreg.fit(X_train_s, y_train)
p_lr_train = logreg.predict_proba(X_train_s)[:, 1]
p_lr_test = logreg.predict_proba(X_test_s)[:, 1]

# ----------------------------------------------------------------------------
# 5. MODEL 2 — Gradient Boosted Trees (challenger model)
# ----------------------------------------------------------------------------
gbm = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=RANDOM_STATE
)
gbm.fit(X_train, y_train)
p_gbm_train = gbm.predict_proba(X_train)[:, 1]
p_gbm_test = gbm.predict_proba(X_test)[:, 1]


# ----------------------------------------------------------------------------
# 6. VALIDATION METRICS (industry standard credit-risk toolkit)
# ----------------------------------------------------------------------------
def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max separation between good/bad score CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def gini(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return float(2 * auc - 1)


def psi(expected, actual, bins=10):
    """Population Stability Index between two score distributions."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)
    e_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    e_counts = np.clip(e_counts, 1e-6, None)
    a_counts = np.clip(a_counts, 1e-6, None)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def score_metrics(y_true, y_score, label):
    return {
        "model": label,
        "auc": round(roc_auc_score(y_true, y_score), 4),
        "gini": round(gini(y_true, y_score), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "brier": round(brier_score_loss(y_true, y_score), 4),
    }


results = []
results.append(score_metrics(y_train, p_lr_train, "LogReg (train)"))
results.append(score_metrics(y_test, p_lr_test, "LogReg (test/holdout)"))
results.append(score_metrics(y_train, p_gbm_train, "GBM (train)"))
results.append(score_metrics(y_test, p_gbm_test, "GBM (test/holdout)"))

# 5-fold stratified CV (stability across resamples — proxy for repeat-sample robustness)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc_lr = cross_val_score(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    scaler.fit_transform(X), y, cv=skf, scoring="roc_auc"
)
cv_auc_gbm = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=RANDOM_STATE),
    X, y, cv=skf, scoring="roc_auc"
)

# Bootstrap 95% CI on test AUC (resampling-based uncertainty estimate)
def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(aucs))

lr_ci = bootstrap_auc_ci(y_test, p_lr_test)
gbm_ci = bootstrap_auc_ci(y_test, p_gbm_test)

# PSI: train score distribution vs test score distribution (development vs holdout stability)
psi_lr = psi(p_lr_train, p_lr_test)
psi_gbm = psi(p_gbm_train, p_gbm_test)

# ----------------------------------------------------------------------------
# 7. "BACKTEST" — actual vs. expected default rate by score decile (holdout)
#    This is the standard bank model-validation backtesting table:
#    rank-order the holdout sample by predicted PD, bucket into deciles,
#    and compare each bucket's average predicted PD to its observed default rate.
# ----------------------------------------------------------------------------
def decile_backtest(y_true, y_score, label):
    d = pd.DataFrame({"y": y_true.values, "score": y_score})
    d["decile"] = pd.qcut(d["score"], 10, labels=False, duplicates="drop")
    tbl = d.groupby("decile").agg(
        n=("y", "size"),
        avg_predicted_pd=("score", "mean"),
        observed_default_rate=("y", "mean"),
    ).reset_index()
    tbl["decile"] = tbl["decile"] + 1
    tbl["model"] = label
    return tbl

bt_lr = decile_backtest(y_test, p_lr_test, "LogReg")
bt_gbm = decile_backtest(y_test, p_gbm_test, "GBM")

# Hosmer-Lemeshow-style chi-square goodness-of-fit on the decile table
def hosmer_lemeshow(tbl):
    obs_events = tbl["n"] * tbl["observed_default_rate"]
    exp_events = tbl["n"] * tbl["avg_predicted_pd"]
    obs_nonevents = tbl["n"] - obs_events
    exp_nonevents = tbl["n"] - exp_events
    chi2 = (((obs_events - exp_events) ** 2 / exp_events.replace(0, np.nan)) +
            ((obs_nonevents - exp_nonevents) ** 2 / exp_nonevents.replace(0, np.nan))).sum()
    dof = len(tbl) - 2
    pval = 1 - stats.chi2.cdf(chi2, dof)
    return float(chi2), int(dof), float(pval)

hl_lr = hosmer_lemeshow(bt_lr)
hl_gbm = hosmer_lemeshow(bt_gbm)

# ----------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE / COEFFICIENTS (model transparency for review)
# ----------------------------------------------------------------------------
lr_coef = pd.Series(logreg.coef_[0], index=feature_cols).sort_values(key=np.abs, ascending=False)
gbm_importance = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ----------------------------------------------------------------------------
# 9. PLOTS
# ----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130})

# ROC curves
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, yt, ys in [("LogReg", y_test, p_lr_test), ("GBM", y_test, p_gbm_test)]:
    fpr, tpr, _ = roc_curve(yt, ys)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yt, ys):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Holdout Test Set")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/roc_curve.png"); plt.close(fig)

# Calibration curves
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, ys in [("LogReg", p_lr_test), ("GBM", p_gbm_test)]:
    frac_pos, mean_pred = calibration_curve(y_test, ys, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Observed default rate")
ax.set_title("Calibration — Holdout Test Set")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/calibration_curve.png"); plt.close(fig)

# Backtest: predicted vs observed by decile
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, tbl, name in zip(axes, [bt_lr, bt_gbm], ["LogReg", "GBM"]):
    ax.plot(tbl["decile"], tbl["avg_predicted_pd"], marker="o", label="Predicted PD")
    ax.plot(tbl["decile"], tbl["observed_default_rate"], marker="s", label="Observed default rate")
    ax.set_title(f"{name} — Backtest by Score Decile")
    ax.set_xlabel("Score decile (1=lowest risk, 10=highest risk)")
axes[0].set_ylabel("Default rate")
axes[0].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/decile_backtest.png"); plt.close(fig)

# Score distribution shift (train vs test) — visual PSI check
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.hist(p_gbm_train, bins=30, alpha=0.5, density=True, label="Train (development)")
ax.hist(p_gbm_test, bins=30, alpha=0.5, density=True, label="Test (holdout)")
ax.set_xlabel("Predicted PD (GBM)"); ax.set_ylabel("Density")
ax.set_title(f"Score Stability: Train vs Holdout (PSI={psi_gbm:.4f})")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/psi_distribution.png"); plt.close(fig)

# Top feature importances (GBM)
fig, ax = plt.subplots(figsize=(6, 5))
top = gbm_importance.head(12).sort_values()
ax.barh(top.index, top.values)
ax.set_title("Top 12 Feature Importances — GBM")
fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png"); plt.close(fig)

# ----------------------------------------------------------------------------
# 10. SAVE ALL NUMERIC RESULTS TO JSON (single source of truth for the report)
# ----------------------------------------------------------------------------
summary = {
    "data_quality_notes": data_quality_notes,
    "train_test_split": {"train_n": int(len(X_train)), "test_n": int(len(X_test)),
                           "train_prevalence": float(y_train.mean()), "test_prevalence": float(y_test.mean())},
    "holdout_metrics": results,
    "cv_auc_logreg": {"mean": round(float(cv_auc_lr.mean()), 4), "std": round(float(cv_auc_lr.std()), 4),
                       "folds": [round(float(v), 4) for v in cv_auc_lr]},
    "cv_auc_gbm": {"mean": round(float(cv_auc_gbm.mean()), 4), "std": round(float(cv_auc_gbm.std()), 4),
                    "folds": [round(float(v), 4) for v in cv_auc_gbm]},
    "bootstrap_auc_ci_holdout": {
        "logreg": {"mean": round(lr_ci[2], 4), "ci95": [round(lr_ci[0], 4), round(lr_ci[1], 4)]},
        "gbm": {"mean": round(gbm_ci[2], 4), "ci95": [round(gbm_ci[0], 4), round(gbm_ci[1], 4)]},
    },
    "psi_train_vs_holdout": {"logreg": round(psi_lr, 4), "gbm": round(psi_gbm, 4)},
    "hosmer_lemeshow": {
        "logreg": {"chi2": round(hl_lr[0], 3), "dof": hl_lr[1], "p_value": round(hl_lr[2], 4)},
        "gbm": {"chi2": round(hl_gbm[0], 3), "dof": hl_gbm[1], "p_value": round(hl_gbm[2], 4)},
    },
    "top_logreg_coefficients": lr_coef.head(10).round(4).to_dict(),
    "top_gbm_importances": gbm_importance.head(10).round(4).to_dict(),
}

with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

bt_lr.to_csv(f"{OUT}/backtest_table_logreg.csv", index=False)
bt_gbm.to_csv(f"{OUT}/backtest_table_gbm.csv", index=False)

print(json.dumps(summary, indent=2))
print("\nDONE. Artifacts written to", OUT)
