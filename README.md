# Credit Risk PD Model

Probability-of-default model built and validated end to end on the UCI "Default of
Credit Card Clients" dataset (30K accounts). Logistic regression baseline vs. gradient
boosted trees challenger, scored with a full credit-risk validation framework:
discrimination (AUC/KS/Gini), calibration (Hosmer-Lemeshow), stability (CV, bootstrap,
PSI), and a decile-level backtest.

| Model | AUC | KS | Calibration |
|---|---|---|---|
| Logistic Regression | 0.752 | 0.395 | Fails — see report §6 |
| Gradient Boosted Trees | 0.782 | 0.431 | Passes |

## Contents
- `Credit_Risk_Model.ipynb` — full executed pipeline, start here
- `Credit_Risk_Validation_Report.docx` — bank-style validation write-up
- `pipeline.py` — same pipeline as a script
- `artifacts/` — charts, backtest tables, `results_summary.json`

## Run it
```
pip install -r requirements.txt
python3 pipeline.py
```

## Scope
Proves the PD build/validation workflow, not a production model — no true out-of-time
backtest, no reject inference, no fair-lending testing, no LGD/EAD (see report §8).
