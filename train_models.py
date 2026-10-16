import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/crimes_for_model.csv")

features = ["hour", "dayofweek", "is_weekend", "latitude", "longitude"]
X = df[features]
y = df["crime_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print()

models = {
    "Baseline (always THEFT)": DummyClassifier(strategy="most_frequent"),
    "Decision Tree (depth 12)": DecisionTreeClassifier(max_depth=12, random_state=42),
    "KNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    "Random Forest (100 trees)": RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
}

rows = []
for name, model in models.items():
    print("Training", name, "...")
    start = time.time()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train.iloc[:20000])  # sample, KNN is slow on all rows

    rows.append({
        "model": name,
        "train_acc": round(accuracy_score(y_train.iloc[:20000], train_pred), 3),
        "test_acc": round(accuracy_score(y_test, test_pred), 3),
        "test_macro_f1": round(f1_score(y_test, test_pred, average="macro"), 3),
        "seconds": round(time.time() - start),
    })

print()
print(pd.DataFrame(rows).to_string(index=False))