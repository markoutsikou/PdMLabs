import pandas as pd

df = pd.read_csv("data/ims.csv")

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== SOURCES ===")
print(df["source"].value_counts().sort_index())

print("\n=== EVENTS ===")
print(df["event"].value_counts(dropna=False))

print("\n=== EVENTS PER SOURCE ===")
print(
    df.groupby("source")["event"]
      .value_counts(dropna=False)
)

print("\n=== TIMESTAMP RANGE PER SOURCE ===")
print(
    df.groupby("source")["Artificial_timestamp"]
      .agg(["first", "last"])
)

print("\n=== MISSING VALUES ===")
missing = df.isna().sum()
print(missing[missing > 0])

print("\n=== NUMBER OF FEATURE COLUMNS ===")
metadata = ["Artificial_timestamp", "event", "source"]
features = [c for c in df.columns if c not in metadata]

print(len(features))
print(features[:10])