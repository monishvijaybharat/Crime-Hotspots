import pandas as pd

df = pd.read_csv("data/chicago_crimes.csv")
df["date"] = pd.to_datetime(df["date"])

df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute
df["dayofweek"] = df["date"].dt.dayofweek      # 0 = Monday, 6 = Sunday
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# File 1: for the hot-spot map (keep ALL rows, location is reliable)
map_df = df[["date", "primary_type", "latitude", "longitude"]]
map_df.to_csv("data/crimes_for_map.csv", index=False)

# File 2: for the crime-type predictor (drop fake-midnight rows)
fake_midnight = (df["hour"] == 0) & (df["minute"] == 0)
model_df = df[~fake_midnight].copy()

top8 = model_df["primary_type"].value_counts().head(8).index
model_df["crime_group"] = model_df["primary_type"].where(
    model_df["primary_type"].isin(top8), "ALL OTHERS"
)

model_df = model_df[["hour", "dayofweek", "is_weekend",
                     "latitude", "longitude", "crime_group"]]
model_df.to_csv("data/crimes_for_model.csv", index=False)

print("Map file rows:  ", len(map_df))
print("Model file rows:", len(model_df))
print()
print("Latitude range: ", df["latitude"].min(), "to", df["latitude"].max())
print("Longitude range:", df["longitude"].min(), "to", df["longitude"].max())
print()
print("Crime groups (share of model data):")
print(model_df["crime_group"].value_counts(normalize=True).round(3))