import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
model = RandomForestClassifier(random_state=42)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی
predictions = model.predict(X_test)

# محاسبه دقت
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)