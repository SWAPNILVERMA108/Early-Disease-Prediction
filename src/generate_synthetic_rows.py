"""
Extend dataset.csv (500 rows) to 5,000 rows with 4,500 synthetic patient records.

Project: Early Disease Prediction Using Machine Learning on Health Data

Design choices
--------------
* Same 8 columns, same order, same dtypes as dataset.csv.
* Feature ranges are taken from the original file (min/max), and features are
  drawn uniformly inside them, so the new rows look like the original ones.
  Smoking is 0/1 with probability 0.5, as in the original file.
* patient_id continues the original numbering: P0501 ... P5000.
* The label is generated from an illustrative risk score in which older age,
  higher BMI, higher blood pressure, higher glucose, higher cholesterol and
  smoking all raise the chance of disease (plus clinical-style thresholds for
  glucose, blood pressure and BMI) and random noise is added.
  This is an ASSUMED relationship, not real patient data.
* The number of positive labels is fixed so that the FULL 5,000-row file is
  exactly 50% disease / 50% no disease (the original 500 rows are 39.4% disease).

Usage:  python3 generate_synthetic_rows.py dataset.csv dataset_5000.csv
"""
import sys

import numpy as np
import pandas as pd

SEED = 42
TOTAL_ROWS = 5000


def main(src: str, out: str) -> None:
    orig = pd.read_csv(src)
    rng = np.random.default_rng(SEED)
    n_new = TOTAL_ROWS - len(orig)

    # ---- features: uniform inside the ranges seen in the original file -------
    lo, hi = orig.min(numeric_only=True), orig.max(numeric_only=True)
    new = pd.DataFrame(
        {
            "age": rng.integers(lo["age"], hi["age"] + 1, n_new),
            "bmi": np.round(rng.uniform(lo["bmi"], hi["bmi"], n_new), 1),
            "blood_pressure": rng.integers(lo["blood_pressure"], hi["blood_pressure"] + 1, n_new),
            "glucose": rng.integers(lo["glucose"], hi["glucose"] + 1, n_new),
            "cholesterol": rng.integers(lo["cholesterol"], hi["cholesterol"] + 1, n_new),
            "smoking": rng.integers(0, 2, n_new),
        }
    )

    # ---- illustrative risk score (assumed, not medically validated) ----------
    z = (
        0.035 * (new["age"] - 50)
        + 0.060 * (new["bmi"] - 28.6)
        + 0.025 * (new["blood_pressure"] - 100)
        + 0.030 * (new["glucose"] - 134)
        + 0.010 * (new["cholesterol"] - 216)
        + 0.60 * (new["smoking"] - 0.5)
        + 0.60 * (new["glucose"] >= 126)        # diabetic-range glucose
        + 0.40 * (new["blood_pressure"] >= 130)  # high blood pressure
        + 0.30 * (new["bmi"] >= 30)              # obesity
    )

    # ---- labels: add noise, then take exactly k positives so the full file is 50/50
    k_positive = TOTAL_ROWS // 2 - int(orig["label"].sum())
    noisy = z + rng.logistic(0.0, 1.0, n_new)
    label = np.zeros(n_new, dtype=int)
    label[np.argsort(-noisy.to_numpy())[:k_positive]] = 1
    new["label"] = label

    # ---- ids + column order + stack ------------------------------------------
    new.insert(0, "patient_id", [f"P{i:04d}" for i in range(len(orig) + 1, TOTAL_ROWS + 1)])
    full = pd.concat([orig, new[orig.columns]], ignore_index=True)
    full.to_csv(out, index=False)
    print(f"wrote {out}: {full.shape[0]} rows x {full.shape[1]} columns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
