import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

### Load dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
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

## Encoding text columns using one hot encoding
preprocess = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols),], remainder="drop")

# Converting training and testing data into encoded format
X_train_encode = preprocess.fit_transform(X_train)
X_test_encode = preprocess.transform(X_test)

def evaluate_model(name, model):
    ## Training the model, prediction, and print results.
    model.fit(X_train_encode, y_train)
    preds = model.predict(X_test_encode)

    print("Model: ", name)
    print("Accuracy: ", accuracy_score(y_test, preds))
    
    ## Creating Confuxion matrix 
    cm = confusion_matrix(y_test, preds)
    ## Displaying confusion matrix as an image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["attack", "normal"])

    print("\nreport:")
    print(classification_report(y_test, preds))

    ## Showing confusion matrix 
    disp.plot(cmap="Blues")
    ## Adding titles
    plt.title(f"Confusion Matrix - {name}")

### Logistic Regression
log_reg = LogisticRegression(max_iter=2000)
evaluate_model("Logistic Regression", log_reg)

## Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

evaluate_model("Random Forest", rf)

plt.show()

