
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "processed" / "train_clean.csv"


def test_cleaned_file_exists():
    assert CLEAN_DATA.exists()


def test_cleaned_data_has_expected_columns():
    df = pd.read_csv(CLEAN_DATA)

    expected_columns = {
        "record_id",
        "date",
        "queue_wait_sec",
        "agent_exp_months",
        "previous_calls",
        "issue_count",
        "hour",
        "caller_age",
        "language",
        "team",
        "handle_min"
    }

    assert expected_columns.issubset(set(df.columns))


def test_cleaned_data_has_no_duplicate_ids():
    df = pd.read_csv(CLEAN_DATA)

    assert df["record_id"].is_unique


def test_cleaned_data_has_no_missing_target():
    df = pd.read_csv(CLEAN_DATA)

    assert df["handle_min"].notna().all()


def test_cleaned_data_has_no_leakage_column():
    df = pd.read_csv(CLEAN_DATA)

    assert "duration_from_phone_bill" not in df.columns


def test_hour_values_are_valid():
    df = pd.read_csv(CLEAN_DATA)

    valid_hours = df["hour"].dropna()

    assert valid_hours.between(0, 23).all()
