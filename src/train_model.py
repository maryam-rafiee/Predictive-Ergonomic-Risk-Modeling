import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# خواندن نام ویژگی‌ها
features = pd.read_csv(
    "data/UCI HAR Dataset/features.txt",
    sep=r"\s+",
    header=None
)

feature_names = features.iloc[:, 1].values

# داده‌های آموزشی
X_train = pd.read_csv(
    "data/UCI HAR Dataset/train/X_train.txt",
    sep=r"\s+",
    header=None
)



y_train = pd.read_csv(
    "data/UCI HAR Dataset/train/y_train.txt",
    header=None
).values.ravel()

# داده‌های آزمون
X_test = pd.read_csv(
    "data/UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
)


y_test = pd.read_csv(
    "data/UCI HAR Dataset/test/y_test.txt",
    header=None
).values.ravel()

# ساخت مدل
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42)


# آموزش مدل
grid.fit(X_train, y_train)
model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# پیش‌بینی
predictions = model.predict(X_test)

# محاسبه دقت
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()
# Feature Importance
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

print("\nTop 10 Most Important Features:\n")

for i in range(10):
    print(f"{i+1}. Feature {indices[i]} Importance: {importances[indices[i]]:.4f}")
# Plot Top 10 Feature Importances
top10 = indices[:10]

plt.figure(figsize=(10,5))
plt.bar(range(10), importances[top10])
plt.xticks(range(10), top10, rotation=45)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Top 10 Most Important Features")
plt.tight_layout()

plt.show()
joblib.dump(model, "results/random_forest_model.pkl")
print("Model saved successfully!")