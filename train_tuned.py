import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/crimes_for_model.csv")
features = ["hour", "dayofweek", "is_weekend", "latitude", "longitude"]
X = df[features]
y = df["crime_group"]

# Same test split as before, so numbers are comparable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Carve a validation set out of the training data, for tuning only
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
)

print("TUNING min_samples_leaf (scored on validation, NOT test)")
results = []
for leaf in [5, 20, 50, 100, 200]:
    m = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf,
                               n_jobs=-1, random_state=42)
    m.fit(X_fit, y_fit)
    results.append({
        "leaf": leaf,
        "train_acc": round(accuracy_score(y_fit, m.predict(X_fit)), 3),
        "val_acc": round(accuracy_score(y_val, m.predict(X_val)), 3),
        "val_macro_f1": round(f1_score(y_val, m.predict(X_val), average="macro"), 3),
    })
res = pd.DataFrame(results)
print(res.to_string(index=False))

best_leaf = int(res.sort_values("val_acc", ascending=False).iloc[0]["leaf"])
print("\nBest min_samples_leaf by validation accuracy:", best_leaf)

# Final model: retrain on all training data, test ONCE
final = RandomForestClassifier(n_estimators=200, min_samples_leaf=best_leaf,
                               n_jobs=-1, random_state=42)
final.fit(X_train, y_train)
pred = final.predict(X_test)

print("\nFINAL RESULTS ON THE TEST SET")
print("Train accuracy:", round(accuracy_score(y_train, final.predict(X_train)), 3))
print("Test accuracy: ", round(accuracy_score(y_test, pred), 3))
print("Test macro F1: ", round(f1_score(y_test, pred, average="macro"), 3))

# Top-3 accuracy
classes = final.classes_
proba = final.predict_proba(X_test)
top3 = np.argsort(proba, axis=1)[:, -3:]
y_idx = np.searchsorted(classes, y_test)
top3_acc = (top3 == y_idx[:, None]).any(axis=1).mean()
base_top3_labels = y_train.value_counts().head(3).index
base_top3 = y_test.isin(base_top3_labels).mean()
print("\nTop-3 accuracy:", round(top3_acc, 3),
      "| baseline (always the 3 most common):", round(base_top3, 3))

print("\nPER-CRIME-GROUP RESULTS:")
print(classification_report(y_test, pred, zero_division=0))

print("FEATURE IMPORTANCE:")
imp = pd.Series(final.feature_importances_, index=features).sort_values(ascending=False)
print(imp.round(3))

joblib.dump(final, "model.joblib")
print("\nSaved model.joblib")