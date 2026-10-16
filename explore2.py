import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/chicago_crimes.csv")
df["date"] = pd.to_datetime(df["date"])

print("Earliest:", df["date"].min())
print("Latest:  ", df["date"].max())
print("Number of crime types:", df["primary_type"].nunique())

top_share = df["primary_type"].value_counts(normalize=True).iloc[0]
print("Baseline (always guess the top crime):", round(top_share * 100, 1), "%")

print()
print("Crimes per hour of day:")
print(df["date"].dt.hour.value_counts().sort_index())

plt.figure(figsize=(8, 10))
plt.scatter(df["longitude"], df["latitude"], s=1, alpha=0.1)
plt.title("Every crime in the dataset")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig("data/crime_scatter.png", dpi=150)
print()
print("Saved data/crime_scatter.png")