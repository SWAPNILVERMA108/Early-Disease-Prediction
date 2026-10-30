"""Loading, validating, cleaning and splitting the health dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_data(path=C.DATASET_PATH) -> pd.DataFrame:
    """Read the CSV file into a DataFrame."""
    return pd.read_csv(path)


def source_of(df: pd.DataFrame) -> pd.Series:
    """'original' for P0001-P0500, 'synthetic' for every later patient id."""
    number = df[C.ID_COL].str[1:].astype(int)
    return pd.Series(
        ["original" if n <= C.N_ORIGINAL_ROWS else "synthetic" for n in number],
        index=df.index,
        name="source",
    )


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic quality checks and return a report (it never changes the data)."""
    expected = [C.ID_COL] + C.FEATURES + [C.TARGET]
    report = {
        "n_rows": int(len(df)),
        "columns_ok": list(df.columns) == expected,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ids": int(df[C.ID_COL].duplicated().sum()) if C.ID_COL in df else -1,
        "duplicate_rows": int(df.duplicated(subset=[c for c in expected if c != C.ID_COL]).sum()),
        "out_of_range": {},
        "label_values_ok": bool(set(df[C.TARGET].unique()) <= {0, 1}) if C.TARGET in df else False,
    }
    for col, (lo, hi) in C.VALID_RANGES.items():
        if col in df:
            report["out_of_range"][col] = int(((df[col] < lo) | (df[col] > hi)).sum())

    issues = []
    if not report["columns_ok"]:
        issues.append("columns differ from the expected list")
    if report["missing_values"]:
        issues.append(f"{report['missing_values']} missing values")
    if report["duplicate_ids"]:
        issues.append(f"{report['duplicate_ids']} duplicate patient ids")
    if report["duplicate_rows"]:
        issues.append(f"{report['duplicate_rows']} duplicate records")
    bad = {k: v for k, v in report["out_of_range"].items() if v}
    if bad:
        issues.append(f"out-of-range values: {bad}")
    if not report["label_values_ok"]:
        issues.append("label contains values other than 0 and 1")
    report["issues"] = issues
    report["passed"] = not issues
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: drop incomplete / duplicate / out-of-range rows, fix dtypes.

    Out-of-range values are dropped (not clipped) so no invented values enter the data.
    On the shipped dataset nothing is removed, because it already passes validation.
    """
    out = df.copy()
    out[C.ID_COL] = out[C.ID_COL].astype(str).str.strip()
    out = out.dropna()
    out = out.drop_duplicates(subset=C.ID_COL, keep="first")
    for col, (lo, hi) in C.VALID_RANGES.items():
        out = out[(out[col] >= lo) & (out[col] <= hi)]
    for col in ["age", "blood_pressure", "glucose", "cholesterol", "smoking", C.TARGET]:
        out[col] = out[col].astype(int)
    out["bmi"] = out["bmi"].astype(float).round(1)
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = C.TEST_SIZE, seed: int = C.RANDOM_STATE):
    """Stratified train/test split.

    The split keeps the label ratio AND the original/synthetic mix the same in both parts.
    """
    strat = df[C.TARGET].astype(str) + "_" + source_of(df)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def xy(df: pd.DataFrame):
    """Split a DataFrame into the feature table X and the target y."""
    return df[C.FEATURES], df[C.TARGET]


def save_processed_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write the exact train and test rows used in the experiments."""
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(C.PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(C.PROCESSED_DIR / "test.csv", index=False)
