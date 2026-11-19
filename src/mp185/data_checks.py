
import pandas as pd

# Load datasets
train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\n========== 1. LEAKING COLUMN ==========")

print("Columns missing from holdout:")
print(set(train.columns) - set(holdout.columns))

if "duration_from_phone_bill" in train.columns:
    print("Correlation with target:")
    print(train["duration_from_phone_bill"].corr(train["handle_min"]))


print("\n========== 2. DUPLICATE ROWS ==========")

print("Duplicate rows:", train.duplicated().sum())
print("Duplicate record IDs:", train["record_id"].duplicated().sum())


print("\n========== 3. SENTINEL VALUES ==========")

for col in train.select_dtypes(include="number").columns:
    print(f"\n{col}")
    print(train[col].value_counts().head(10))


print("\n========== 4. UNIT MIXUP ==========")

for col in train.select_dtypes(include="number").columns:
    print(f"\n{col}")
    print(train.groupby("team")[col].agg(["min", "mean", "max"]))


print("\n========== 5. STUCK SENSOR ==========")

for col in train.select_dtypes(include="number").columns:
    print(f"\n{col}")
    print("Unique values:", train[col].nunique())
    print("Most common values:")
    print(train[col].value_counts().head(5))


print("\n========== 6. TARGET TYPOS ==========")

print(train["handle_min"].describe())
print("Largest target values:")
print(train.nlargest(15, "handle_min")[["record_id", "handle_min"]])


print("\n========== 7. SPELLING VARIANTS ==========")

for col in ["language", "team"]:
    print(f"\n{col} in train:")
    print(train[col].value_counts(dropna=False))

    print(f"\n{col} in later:")
    print(later[col].value_counts(dropna=False))

print("\n========== DEEP INVESTIGATION ==========")

# A. Sentinel value count
print("\n--- HOUR = 9999 ---")
print("Count:", (train["hour"] == 9999).sum())
print(train.loc[train["hour"] == 9999,
                ["record_id", "date", "hour"]])

# B. Unit mixup verification
print("\n--- QUEUE WAIT BY TEAM ---")
print(train.groupby("team")["queue_wait_sec"].describe())

gamma = train[train["team"] == "Team Gamma"].copy()
gamma["converted_wait_sec"] = gamma["queue_wait_sec"] * 60

print("\nGamma after multiplying by 60:")
print(gamma["converted_wait_sec"].describe())

# C. Target typo investigation
print("\n--- LARGE TARGET VALUES ---")
print(train[train["handle_min"] > 40][
    ["record_id", "date", "team",
     "issue_count", "handle_min"]
].sort_values("handle_min", ascending=False))

# D. Stuck sensor investigation
print("\n--- CONSECUTIVE REPEATED VALUES ---")

numeric_cols = train.select_dtypes(include="number").columns

sorted_train = train.sort_values("date")

for col in numeric_cols:
    values = sorted_train[col].tolist()

    max_run = 1
    current_run = 1

    for i in range(1, len(values)):
        if values[i] == values[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    print(col, "Maximum consecutive repetition:", max_run)

# E. Category verification
print("\n--- ALL CATEGORY VALUES ---")

for col in ["language", "team"]:
    print(col, train[col].unique())
