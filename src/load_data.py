import pandas as pd

# نام ویژگی‌ها
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

# برچسب‌های آموزشی
y_train = pd.read_csv(
    "data/UCI HAR Dataset/train/y_train.txt",
    header=None
)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("\nFirst 5 rows:")
print(X_train.head())