import joblib
import pandas as pd

# Load trained model
model = joblib.load("results/random_forest_model.pkl")

# Load one sample from test dataset
X_test = pd.read_csv(
    "data/UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
)

# Predict first sample
prediction = model.predict(X_test.iloc[[0]])

print("Predicted Activity:", prediction[0])