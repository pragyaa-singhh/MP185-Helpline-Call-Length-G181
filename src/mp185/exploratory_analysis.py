from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "reports" / "eda"
PLOTS = OUT / "plots"

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

with open(OUT / "data_profile.txt", "w", encoding="utf-8") as f:
    f.write("HELPLINE CALL LENGTH - EDA REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Data Types:\n" + str(df.dtypes) + "\n\n")
    f.write("Missing Values:\n" + str(df.isnull().sum()) + "\n\n")
    f.write("Statistics:\n" + str(df.describe(include="all")) + "\n")
    f.write(f"\nDuplicate Rows: {df.duplicated().sum()}\n")

sns.set_theme(style="whitegrid")
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(PLOTS / f"{col}_histogram.png", dpi=150)
    plt.close()

if "handle_min" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["handle_min"].dropna(), kde=True)
    plt.title("Target Distribution - Handle Minutes")
    plt.xlabel("Handle Minutes")
    plt.tight_layout()
    plt.savefig(PLOTS / "target_distribution.png", dpi=150)
    plt.close()

if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS / "correlation_matrix.png", dpi=150)
    plt.close()

if "handle_min" in df.columns:
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() <= 20:
            plt.figure(figsize=(9, 5))
            sns.boxplot(data=df, x=col, y="handle_min")
            plt.xticks(rotation=30)
            plt.title(f"Handle Time by {col}")
            plt.tight_layout()
            plt.savefig(PLOTS / f"handle_min_by_{col}.png", dpi=150)
            plt.close()

print("EDA completed successfully!")
print("Data shape:", df.shape)
print("Profile:", OUT / "data_profile.txt")
print("Graphs:", PLOTS)
