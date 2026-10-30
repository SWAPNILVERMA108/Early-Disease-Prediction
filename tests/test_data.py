"""Checks on the dataset files and the data helper functions."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data_utils as D


@pytest.fixture(scope="module")
def df():
    return D.load_data()


def test_shape_and_column_order(df):
    assert df.shape == (5000, 8)
    assert list(df.columns) == [C.ID_COL] + C.FEATURES + [C.TARGET]


def test_ids_are_unique_and_sequential(df):
    assert df[C.ID_COL].is_unique
    assert df[C.ID_COL].tolist() == [f"P{i:04d}" for i in range(1, 5001)]


def test_first_500_rows_are_the_untouched_original(df):
    original = pd.read_csv(C.ORIGINAL_PATH)
    pd.testing.assert_frame_equal(df.iloc[:500].reset_index(drop=True), original)


def test_dataset_passes_validation(df):
    report = D.validate_data(df)
    assert report["passed"], report["issues"]
    assert report["missing_values"] == 0


def test_labels_are_exactly_balanced(df):
    assert df[C.TARGET].value_counts().to_dict() == {0: 2500, 1: 2500}


def test_validation_catches_bad_rows(df):
    bad = df.head(10).copy()
    bad.loc[0, "age"] = 200                       # impossible age
    bad.loc[1, "glucose"] = np.nan                # missing value
    bad.loc[2, C.ID_COL] = bad.loc[3, C.ID_COL]   # duplicate patient id
    report = D.validate_data(bad)
    assert not report["passed"]
    assert report["out_of_range"]["age"] == 1
    assert report["missing_values"] == 1
    assert report["duplicate_ids"] == 1


def test_clean_data_removes_bad_rows_and_keeps_good_ones(df):
    bad = df.head(10).copy()
    bad.loc[0, "age"] = 200
    bad.loc[1, "glucose"] = np.nan
    bad.loc[2, C.ID_COL] = bad.loc[3, C.ID_COL]
    cleaned = D.clean_data(bad)
    assert len(cleaned) == 7
    assert D.validate_data(cleaned)["passed"]


def test_clean_data_changes_nothing_on_the_shipped_file(df):
    pd.testing.assert_frame_equal(D.clean_data(df), df, check_dtype=False)


def test_source_of():
    ids = pd.DataFrame({C.ID_COL: ["P0001", "P0500", "P0501", "P5000"]})
    assert D.source_of(ids).tolist() == ["original", "original", "synthetic", "synthetic"]


def test_split_sizes_stratification_and_no_overlap(df):
    train, test = D.split_data(df)
    assert (len(train), len(test)) == (4000, 1000)
    assert set(train[C.ID_COL]).isdisjoint(test[C.ID_COL])
    assert train[C.TARGET].mean() == pytest.approx(0.5, abs=0.002)
    assert test[C.TARGET].mean() == pytest.approx(0.5, abs=0.002)
    # the original/synthetic mix is also kept: 10% of both parts are original rows
    assert (D.source_of(train) == "original").mean() == pytest.approx(0.1, abs=0.005)
    assert (D.source_of(test) == "original").mean() == pytest.approx(0.1, abs=0.01)


def test_split_is_reproducible(df):
    a, _ = D.split_data(df)
    b, _ = D.split_data(df)
    pd.testing.assert_frame_equal(a, b)


def test_saved_splits_match_a_fresh_split(df):
    train, test = D.split_data(df)
    pd.testing.assert_frame_equal(pd.read_csv(C.PROCESSED_DIR / "train.csv"), train)
    pd.testing.assert_frame_equal(pd.read_csv(C.PROCESSED_DIR / "test.csv"), test)


def test_data_dictionary_describes_every_column():
    dd = pd.read_csv(C.DICTIONARY_PATH)
    assert dd["column_name"].tolist() == [C.ID_COL] + C.FEATURES + [C.TARGET]
    assert not dd["description_fill_in"].str.contains("Learner to fill in").any()
