import pandas as pd
from urllib.parse import urlencode, quote

base = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

params = {
    "$where": "date >= '2023-01-01T00:00:00' AND latitude IS NOT NULL",
    "$order": "date DESC",
    "$limit": 200000,
}

url = base + "?" + urlencode(params, quote_via=quote)

print("Downloading... this can take 30-60 seconds")
df = pd.read_csv(url)
df.to_csv("data/chicago_crimes.csv", index=False)
print("Saved", len(df), "rows to data/chicago_crimes.csv")