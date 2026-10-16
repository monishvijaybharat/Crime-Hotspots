import pandas as pd
import folium
from folium.plugins import HeatMap

df = pd.read_csv("data/crimes_for_map.csv")

# ---- Part 1: ranked grid hot-spots (uses ALL rows) ----
df["cell_lat"] = df["latitude"].round(2)
df["cell_lon"] = df["longitude"].round(2)

cells = (
    df.groupby(["cell_lat", "cell_lon"])
      .size()
      .sort_values(ascending=False)
      .head(10)
)
print("TOP 10 HOTTEST GRID CELLS (crimes per ~1 km square):")
print(cells)
print()

# ---- Part 2: interactive heatmap (sampled for speed) ----
sample = df.sample(50000, random_state=42)

m = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
)
HeatMap(sample[["latitude", "longitude"]].values.tolist(),
        radius=8, blur=12).add_to(m)
m.save("data/hotspot_map.html")
print("Saved data/hotspot_map.html")