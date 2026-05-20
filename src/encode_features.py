import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Loading dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR/"KDDTrain+.txt"

df = pd.read_csv(TRAIN_FILE, header=None)

## IN the dataset
attack_label_col = df.columns[-2]
difficulty_col = df.columns[-1]

## removing difficulty columns
df = df.drop(columns=[difficulty_col])

## Making Normal and Attack labels in binary 
df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")

## Splitting the dataset into features and labels
X = df.drop(columns=[attack_label_col, "binary_label"])
Y = df["binary_label"]

## Finding text and numeric columns
Cat_columns = X.select_dtypes(include="object").columns.tolist()
Num_columns = X.select_dtypes(exclude="object").columns.tolist()

print("Text columns: ", Cat_columns)
print("Number of numeric columns: ", len(Num_columns))

## Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

## One hot encoding for text columns
preprocess = ColumnTransformer(transformers=[("category", OneHotEncoder(handle_unknown="ignore"), Cat_columns), ("numbers", "passthrough", Num_columns),], remainder="drop")

## Only fitted for training data
X_train_encoded = preprocess.fit_transform(X_train)
X_test_encoded = preprocess.transform(X_test)

print("\nOriginal shape of X_train:", X_train.shape)
print("Encoded shape of X_train:", X_train_encoded.shape)
print("\nOriginal Shape of X_test:", X_test.shape)
print("Encoded shape of X_test:", X_test_encoded.shape)