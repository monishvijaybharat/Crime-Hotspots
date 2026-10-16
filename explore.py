import pandas as pd

df = pd.read_csv("data/chicago_crimes.csv")

print("SHAPE (rows, columns):", df.shape)
print()
print("COLUMNS:")
print(df.columns.tolist())
print()
print("FIRST 5 ROWS:")
print(df.head())
print()
print("TOP 10 CRIME TYPES:")
print(df["primary_type"].value_counts().head(10))
print()
print("MISSING VALUES PER COLUMN:")
print(df.isna().sum())