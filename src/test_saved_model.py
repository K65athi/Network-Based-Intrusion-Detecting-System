import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

## Load the saved model and preprocessing
preprocess = joblib.load(MODELS_DIR / "preprocess.pkl")
model = joblib.load(MODELS_DIR / "random_forest_model.pkl")

print("Preprocessing and Model have been loaded successfully.")

## Load dataset
TEST_FILE = DATA_DIR / "KDDTest+.txt"

df = pd.read_csv(TEST_FILE, header=None)

attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

## Removing difficulty column
df = df.drop(columns=[difficulty_col])

## Making Normal and Attack labels in binary
df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")

## Splitting the dataset into features(X) and labels(Y)
X = df.drop(columns=[attack_label_col, "binary_label"])
Y = df["binary_label"]

## Taking a sample of the test data for quick testing
sample = X.iloc[[0]]

actual_label = Y.iloc[0]

print("\nActual Label:", actual_label)

## Preprocess the sample data
sample_encoded = preprocess.transform(sample)

## Making prediction
prediction = model.predict(sample_encoded)

print("Predicted Label:", prediction[0])