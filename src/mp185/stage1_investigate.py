"""
Stage 1: Data Inspection & Finding Data Problems
MP185 – Helpline Call Length Prediction | Group G181
Author: Anya Yadav (S24CSEU2249)

This script investigates all 7 required data-quality problems and
prints detailed evidence for each finding.
Run from repo root:  python src/mp185/stage1_investigate.py
"""

import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
train   = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("=" * 65)
print("STAGE 1 – DATA INVESTIGATION  |  MP185 / G181")
print("=" * 65)
print(f"\nTrain shape   : {train.shape}")
print(f"Holdout shape : {holdout.shape}")
print(f"Train columns : {train.columns.tolist()}")

# ── 1. LEAKING COLUMN ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. LEAKING COLUMN")
print("=" * 65)

train_only = set(train.columns) - set(holdout.columns) - {"handle_min"}
print(f"Columns in train but NOT in holdout (excluding target): {train_only}")

col_leak = "duration_from_phone_bill"
if col_leak in train.columns:
    corr = train[col_leak].corr(train["handle_min"])
    print(f"\nCorrelation of '{col_leak}' with 'handle_min': {corr:.6f}")
    print("Sample rows (first 5):")
    print(train[[col_leak, "handle_min"]].head(5).to_string())
    print("\nConclusion: correlation ~1.0 → this column is a direct leak of the")
    print("target. It is computed AFTER the call ends (phone bill duration).")
    print("It must NOT be used as a feature.")

# ── 2. UNIT MIXUP ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. UNIT MIXUP  (queue_wait_sec by team)")
print("=" * 65)

grp = train.groupby("team")["queue_wait_sec"]
print(grp.describe().round(2).to_string())
print()

for t in sorted(train["team"].unique()):
    sub = train[train["team"] == t]["queue_wait_sec"]
    sample = sub.head(8).tolist()
    print(f"{t}:  min={sub.min():.2f}  mean={sub.mean():.2f}  "
          f"max={sub.max():.2f}  sample={sample}")

print()
print("Observation: Team Gamma values are ~0–10 (minutes), while")
print("Team Alpha & Beta values are ~100–600 (seconds).")
print("Team Gamma recorded queue_wait in MINUTES; others used SECONDS.")
print(f"Conversion factor to harmonise Gamma to seconds: 60 (multiply by 60)")
print(f"Or to convert to consistent unit: factor = 1/60 ≈ 0.0167 for others")

gamma_wait = train[train["team"] == "Team Gamma"]["queue_wait_sec"]
other_wait = train[train["team"] != "Team Gamma"]["queue_wait_sec"]
print(f"\nGamma  queue_wait_sec median : {gamma_wait.median():.2f}")
print(f"Others queue_wait_sec median : {other_wait.median():.2f}")

# ── 3. SENTINEL VALUES ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. SENTINEL / INVALID VALUES")
print("=" * 65)

print("Value counts for 'hour' (top 15):")
print(train["hour"].value_counts().head(15).to_string())

sentinel_val = 9999
sentinel_count = (train["hour"] == sentinel_val).sum()
print(f"\nRows where hour == 9999: {sentinel_count}")
print("Sample rows:")
print(train[train["hour"] == sentinel_val][["record_id", "date", "hour"]].head(5).to_string())
print("\nValid hour range for a call-centre: 8–21 (business hours).")
print("9999 is clearly a placeholder / error code, not a real hour.")

# ── 4. DUPLICATE ROWS ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("4. DUPLICATE ROWS")
print("=" * 65)

dup_full = train.duplicated().sum()
dup_id   = train["record_id"].duplicated().sum()
print(f"Fully identical duplicate rows : {dup_full}")
print(f"Duplicate record_id values     : {dup_id}")

dup_ids = train[train["record_id"].duplicated(keep=False)].sort_values("record_id")
print(f"\nTotal rows involved in record_id duplicates: {len(dup_ids)}")
print("Sample duplicate record_ids:")
print(dup_ids[["record_id", "date", "team", "handle_min"]].head(10).to_string())

# ── 5. STUCK SENSOR ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5. STUCK SENSOR / CONSTANT VALUES")
print("=" * 65)

sorted_train = train.sort_values("date").reset_index(drop=True)
check_cols = ["queue_wait_sec", "agent_exp_months", "previous_calls",
              "issue_count", "caller_age"]

for col in check_cols:
    vals = sorted_train[col].tolist()
    max_run = 1
    cur_run = 1
    best_start = 0
    run_start  = 0
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
                best_start = run_start
        else:
            cur_run   = 1
            run_start = i

    flag = " *** SUSPICIOUS ***" if max_run >= 10 else ""
    print(f"{col}: max consecutive repeats = {max_run}{flag}")
    if max_run >= 10:
        rep_val   = vals[best_start]
        rep_dates = sorted_train["date"].iloc[best_start: best_start + max_run].tolist()
        print(f"  Stuck value: {rep_val}")
        print(f"  Dates ({len(rep_dates)} days): {rep_dates[0]} → {rep_dates[-1]}")

# ── 6. TARGET TYPOS / SCALE ERRORS ───────────────────────────────────────────
print("\n" + "=" * 65)
print("6. TARGET TYPOS / 10× SCALE ERRORS  (handle_min)")
print("=" * 65)

print("Target distribution:")
print(train["handle_min"].describe().round(2).to_string())
print()

normal_max = train["handle_min"].quantile(0.95)
threshold  = normal_max * 3          # generous threshold
large = train[train["handle_min"] > threshold].sort_values("handle_min", ascending=False)
print(f"95th-percentile of handle_min : {normal_max:.2f}")
print(f"Flagging rows > 3× P95 ({threshold:.2f}) as suspected typos/scale errors:")
print(f"Count : {len(large)}")
print(large[["record_id", "date", "team", "issue_count", "handle_min"]].to_string())

# How many rows exceed 10× the median?
median_target = train["handle_min"].median()
typo_10x = (train["handle_min"] > median_target * 10).sum()
print(f"\nMedian handle_min : {median_target:.2f}")
print(f"Rows > 10× median ({median_target * 10:.2f}) : {typo_10x}")

# ── 7. SPELLING VARIANTS ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("7. SPELLING / CATEGORY INCONSISTENCIES")
print("=" * 65)

for col in ["language", "team"]:
    print(f"\n--- {col} in TRAIN ---")
    print(train[col].value_counts(dropna=False).to_string())
    print(f"--- {col} in HOLDOUT ---")
    print(holdout[col].value_counts(dropna=False).to_string())

    train_cats   = set(train[col].dropna().str.strip().str.lower().unique())
    holdout_cats = set(holdout[col].dropna().str.strip().str.lower().unique())
    extras_train  = train_cats   - holdout_cats
    extras_holdout= holdout_cats - train_cats
    if extras_train:
        print(f"  Values only in TRAIN (potential spelling variant): {extras_train}")
    if extras_holdout:
        print(f"  Values only in HOLDOUT : {extras_holdout}")
    if not extras_train and not extras_holdout:
        print("  No spelling differences found between train and holdout.")

print("\n" + "=" * 65)
print("INVESTIGATION COMPLETE")
print("=" * 65)
