import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

## Load dataset
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"

df = pd.read_csv(TRAIN_FILE, header=None)

attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

## Removing difficulty column
df = df.drop(columns=[difficulty_col])

## Making Normal and Attack labels in binary
df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")

## Splitting the dataset into features(X) and labels(Y)
X = df.drop(columns=[attack_label_col, "binary_label"])
Y = df["binary_label"]

# Finding Numeric ind text columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

## Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

## preprocessing using one hot encoding
preprocess = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)])

X_train_encoded = preprocess.fit_transform(X_train)

## Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

## Train the model
model.fit(X_train_encoded, y_train)

## Saving the model and presprocessing
joblib.dump(model, MODELS_DIR / "random_forest_model.pkl")

joblib.dump(preprocess, MODELS_DIR / "preprocess.pkl")

print("Model and preprocessing saved successfully.")