import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("model.joblib")

FEATURES = ["hour", "dayofweek", "is_weekend", "latitude", "longitude"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

LAT_RANGE = (41.64, 42.03)
LON_RANGE = (-87.94, -87.52)

PLACES = {
    "The Loop (downtown)": (41.8819, -87.6278),
    "Navy Pier": (41.8917, -87.6086),
    "Lincoln Park": (41.9214, -87.6513),
    "Hyde Park": (41.7943, -87.5907),
    "Midway Airport": (41.7868, -87.7522),
    "O'Hare Airport": (41.9742, -87.9073),
}

# ---- Crime points for the map page (loaded once at startup) ----
crimes = pd.read_csv("data/crimes_for_map.csv")
crimes["date"] = pd.to_datetime(crimes["date"])
crimes["hour"] = crimes["date"].dt.hour
crimes["dow"] = crimes["date"].dt.dayofweek   # 0 = Monday

# Exactly 00:00 is a placeholder time, so mark the hour as unknown (-1)
fake_midnight = (crimes["hour"] == 0) & (crimes["date"].dt.minute == 0)
crimes.loc[fake_midnight, "hour"] = -1

top8 = crimes["primary_type"].value_counts().head(8).index
crimes["group"] = crimes["primary_type"].where(
    crimes["primary_type"].isin(top8), "ALL OTHERS"
)
CRIME_POINTS = {
    group: sub[["latitude", "longitude", "hour", "dow"]]
           .round({"latitude": 4, "longitude": 4}).values.tolist()
    for group, sub in crimes.groupby("group")
}

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    error = None
    form = {"day": "0", "hour": "15", "place": "The Loop (downtown)",
            "lat": "", "lon": ""}

    if request.method == "POST":
        form.update(request.form.to_dict())
        try:
            hour = int(form["hour"])
            day = int(form["day"])
            if form["lat"].strip() and form["lon"].strip():
                lat = float(form["lat"])
                lon = float(form["lon"])
            else:
                lat, lon = PLACES[form["place"]]
        except (ValueError, KeyError):
            error = "Please enter numbers for the coordinates."
        else:
            inside = (LAT_RANGE[0] <= lat <= LAT_RANGE[1]
                      and LON_RANGE[0] <= lon <= LON_RANGE[1])
            if not inside:
                error = "Those coordinates are outside Chicago."
            else:
                row = pd.DataFrame([[hour, day, int(day >= 5), lat, lon]],
                                   columns=FEATURES)
                proba = model.predict_proba(row)[0]
                top3 = proba.argsort()[::-1][:3]
                results = [(model.classes_[i].title(), round(proba[i] * 100, 1))
                           for i in top3]

    return render_template("index.html", results=results, error=error,
                           form=form, days=DAYS, places=list(PLACES))


@app.route("/map")
def crime_map():
    return render_template("map.html")


@app.route("/api/crimes")
def api_crimes():
    return jsonify(CRIME_POINTS)


if __name__ == "__main__":
    app.run(debug=True, port=5001)