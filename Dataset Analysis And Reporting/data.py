from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("VIRAT KOHLI IPL ANALYSIS")
print("=" * 60)

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "Virat_Kohli_RCB_Dashboard_Dataset_5000_Rows.csv"
output_dir = base_dir / "output"
plots_dir = output_dir / "graphs"
output_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

print("Loading Data...")
df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates().copy()
df.columns = [col.strip() for col in df.columns]

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

for col in ["Runs", "Balls", "Fours", "Sixes", "Season"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["Strike_Rate"] = np.where(df["Balls"] == 0, 0, (df["Runs"] / df["Balls"]) * 100)
df["Boundary_Runs"] = df["Fours"] * 4 + df["Sixes"] * 6
df["Running_Runs"] = (df["Runs"] - df["Boundary_Runs"]).clip(lower=0)

conditions = [df["Runs"] >= 100, df["Runs"] >= 50, df["Runs"] >= 30]
choices = ["Century", "Half Century", "Good"]
df["Performance"] = np.select(conditions, choices, default="Low")

print("\nSeason Wise Runs")
print(df.groupby("Season")["Runs"].sum())

plt.figure(figsize=(8, 4))
df.groupby("Season")["Runs"].sum().plot(kind="bar")
plt.title("Runs by Season")
plt.tight_layout()
plt.savefig(plots_dir / "runs_by_season.png")
plt.close()

plt.figure(figsize=(8, 4))
df.groupby("Opponent")["Runs"].sum().sort_values().plot(kind="barh")
plt.title("Runs vs Opponent")
plt.tight_layout()
plt.savefig(plots_dir / "runs_vs_opponent.png")
plt.close()

plt.figure(figsize=(6, 4))
df["Runs"].hist(bins=20)
plt.title("Runs Distribution")
plt.tight_layout()
plt.savefig(plots_dir / "runs_distribution.png")
plt.close()

df.to_csv(output_dir / "Cleaned_Data.csv", index=False)

with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows : {len(df)}\n")
    f.write(f"Total Runs : {df['Runs'].sum()}\n")
    f.write(f"Average Runs : {df['Runs'].mean():.2f}\n")
    f.write(f"Highest Score : {df['Runs'].max()}\n")
    f.write(f"Average Strike Rate : {df['Strike_Rate'].mean():.2f}\n")

print("Project Completed Successfully")
