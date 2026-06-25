import pandas as pd

features = pd.read_csv(
    "data/UCI HAR Dataset/features.txt",
    sep=r"\s+",
    header=None
)

print(features.head())
print("Number of features:", len(features))