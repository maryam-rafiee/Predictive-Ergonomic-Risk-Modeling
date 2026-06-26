import pandas as pd

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

X_train.columns = feature_names

y_train = pd.read_csv(
    "data/UCI HAR Dataset/train/y_train.txt",
    header=None
)

# داده‌های آزمون
X_test = pd.read_csv(
    "data/UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
)

X_test.columns = feature_names

y_test = pd.read_csv(
    "data/UCI HAR Dataset/test/y_test.txt",
    header=None
)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

print("\nFirst 5 rows:")
print(X_train.head())