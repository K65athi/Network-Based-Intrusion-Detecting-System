import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"

df = pd.read_csv(TRAIN_FILE, header=None)

attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

df = df.drop(columns=[difficulty_col])

df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")

x = df.drop(columns=[attack_label_col, "binary_label"])
y = df["binary_label"]

print("Feature data (X) shape:", x.shape)
print("Label data (y) shape:", y.shape)

print("\nSample features:")
print(x.head(3))

print("\nSample labels:")
print(y.head(10))