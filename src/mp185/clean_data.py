
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    print("\nProcessing:", input_path.name)
    print("Original shape:", df.shape)

    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. Remove duplicate record IDs
    if "record_id" in df.columns:
        df = df.drop_duplicates(subset="record_id", keep="first")

    # 3. Remove data leakage column
    if "duration_from_phone_bill" in df.columns:
        df = df.drop(columns=["duration_from_phone_bill"])

    # 4. Correct Team Gamma unit mix-up
    if "team" in df.columns and "queue_wait_sec" in df.columns:
        gamma = (
            df["team"].astype(str).str.strip().str.lower()
            == "team gamma"
        )
        df.loc[gamma, "queue_wait_sec"] *= 60

    # 5. Handle invalid hour sentinel
    if "hour" in df.columns:
        df.loc[df["hour"] == 9999, "hour"] = pd.NA

    # 6. Correct target values approximately 10x too large
    if (
        "handle_min" in df.columns
        and "duration_from_phone_bill" in pd.read_csv(input_path, nrows=1).columns
    ):
        original = pd.read_csv(input_path)

        original = original.drop_duplicates(
            subset="record_id", keep="first"
        )

        ratio = (
            original["handle_min"]
            / original["duration_from_phone_bill"]
        )

        typo_ids = original.loc[
            ratio.between(8, 12), "record_id"
        ]

        if "record_id" in df.columns:
            mask = df["record_id"].isin(typo_ids)
            df.loc[mask, "handle_min"] /= 10

    # 7. Standardize categorical text
    for col in ["language", "team"]:
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip().str.lower()
            )

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("Cleaned shape:", df.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":

    # Clean training dataset
    clean_data(
        DATA_DIR / "train.csv",
        OUTPUT_DIR / "train_clean.csv"
    )

    # Clean later dataset
    clean_data(
        DATA_DIR / "later.csv",
        OUTPUT_DIR / "later_clean.csv"
    )
