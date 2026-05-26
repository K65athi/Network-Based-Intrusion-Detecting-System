import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

### Load TRAINING dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

def preprocess(df):
    attack_label_col = df.columns[-2]
    difficulty_col = df.columns[-1]
    df = df.drop(columns=[difficulty_col])
    df["binary_label"] = df[attack_label_col].apply(lambda x: "normal" if x == "normal" else "attack")
    X = df.drop(columns=[attack_label_col, "binary_label"])
    Y = df["binary_label"]
    return X, Y

## Load train and test separately
df_train = pd.read_csv(TRAIN_FILE, header=None)
df_test = pd.read_csv(TEST_FILE, header=None)

X_train, y_train = preprocess(df_train)
X_test, y_test = preprocess(df_test)

# Finding Numeric and text columns
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

## Encoding using one hot encoding
encoder = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), 
    ("num", "passthrough", num_cols)
])

# Fit ONLY on training data, transform both
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

def evaluate_model(name, model):
    model.fit(X_train_encoded, y_train)

    preds = model.predict(X_test_encoded)

    ## train data prediction
    train_preds = model.predict(X_train_encoded)

    print("Model:", name)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print("Training Accuracy:", accuracy_score(y_train, train_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    cm_train = confusion_matrix(y_train, train_preds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["attack", "normal"])
    disp_train.plot(cmap="Reds")
    plt.title(f"Confusion Matrix (Train) - {name}")#

    cm_test = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["attack", "normal"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Test) - {name}")


### Logistic Regression
log_reg = LogisticRegression(max_iter=2000)
evaluate_model("Logistic Regression", log_reg)

## Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
evaluate_model("Random Forest", rf)

plt.show()

