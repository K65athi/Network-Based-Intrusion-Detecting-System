import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
df = pd.read_csv(TRAIN_FILE, header=None)

print("Loaded:", TRAIN_FILE.name)
print("Shape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nAttack label distribution (second-to-last column):")
print(df.iloc[:, -2].value_counts().head(10))