import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"

df = pd.read_csv(TRAIN_FILE, header=None)

# In NSL-KDD:
attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

# Remove difficulty column
df = df.drop(columns=[difficulty_col])

# Create binary labels: normal vs attack
df["binary_label"] = df[attack_label_col].apply(
    lambda x: "normal" if x == "normal" else "attack"
)

# -----------------------------
# Split into X (features) and y (label)
# -----------------------------
X = df.drop(columns=[attack_label_col, "binary_label"])
y = df["binary_label"]

# -----------------------------
# Find text columns (categorical) and number columns
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Text columns (need encoding):", cat_cols)
print("Number columns count:", len(num_cols))

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Encode text columns using One-Hot Encoding
# -----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"  # drop anything not listed (safe)
)

# Fit on training data only, then transform both
X_train_encoded = preprocess.fit_transform(X_train)
X_test_encoded = preprocess.transform(X_test)

print("\nOriginal X_train shape:", X_train.shape)
print("Encoded X_train shape:", X_train_encoded.shape)

print("\nOriginal X_test shape:", X_test.shape)
print("Encoded X_test shape:", X_test_encoded.shape)


