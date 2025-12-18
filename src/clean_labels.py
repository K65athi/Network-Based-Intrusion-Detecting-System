import pandas as pd 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"

df = pd.read_csv(TRAIN_FILE, header=None)

print("Original shape (rows, columns):", df.shape)

attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

df = df.drop(columns=[difficulty_col])

print("Shape after removing difficulty column:", df.shape)

df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")

print("\nBinary label counts:")
print(df["binary_label"].value_counts())

# Display some sample rows to verify changes
print("\nSample rows with new binary label:")
print(df[[attack_label_col, "binary_label"]].head(10))