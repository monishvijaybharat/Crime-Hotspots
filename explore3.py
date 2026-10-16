import pandas as pd

df = pd.read_csv("data/chicago_crimes.csv")
df["date"] = pd.to_datetime(df["date"])

df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute

exact_midnight = (df["hour"] == 0) & (df["minute"] == 0)
print("Rows at exactly 00:00:", exact_midnight.sum())
print("Share of all rows:", round(exact_midnight.mean() * 100, 1), "%")
print()

print("Crime mix at exactly 00:00 (top 8):")
print(df[exact_midnight]["primary_type"].value_counts(normalize=True).head(8).round(3))
print()

print("Crime mix everywhere else (top 8):")
print(df[~exact_midnight]["primary_type"].value_counts(normalize=True).head(8).round(3))
print()

print("Hour 0 WITHOUT the exact-midnight rows:")
print(((df["hour"] == 0) & ~exact_midnight).sum())
print("Hour 1 for comparison:")
print((df["hour"] == 1).sum())