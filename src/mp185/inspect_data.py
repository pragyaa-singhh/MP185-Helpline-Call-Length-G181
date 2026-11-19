
import pandas as pd

# Load training data
df = pd.read_csv("data/raw/train.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== NUMERIC SUMMARY ==========")
print(df.describe().T)

print("\n========== CATEGORICAL VALUES ==========")
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\n========== FIRST 5 ROWS ==========")
print(df.head())
