import tkinter as tk
from tkinter import messagebox
import pandas as pd
from pathlib import Path
import joblib

## Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

## Load the saved model and preprocessing
preprocess = joblib.load(MODELS_DIR / "preprocess.pkl")
model = joblib.load(MODELS_DIR / "random_forest_model.pkl")

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

## Function to make prediction 
def predict():
    sample = X.sample(n=5)
    actual_label = Y.loc[sample.index]
    
    sample_encoded = preprocess.transform(sample)

    prediction = model.predict(sample_encoded)
    ## Output text
    output = ""

    for i in range(len(sample)):
        output += (f"Sample {i+1}\n" f"Actual Label: {actual_label.iloc[i]}\n" f"Prediction: {prediction[i]}\n\n")

    ## Update the result text in the GUI
    result_text.set(output)

## Create the main application window
app = tk.Tk()
app.title("ML Based Intrusion Detection System")

app.geometry("700x700")

## Title Label
title_label = tk.Label(app, text="ML Based Intrusion Detection System", font=("Arial", 16, "bold"))
title_label.pack(pady=20)

## Result Label
result_text = tk.StringVar()
result_text.set("Click the button to test the prediction")

result_label = tk.Label(app, textvariable=result_text, font=("Arial", 14))
result_label.pack(pady=20)

## Prediction Button
prediction_button = tk.Button(app, text="Predict Sample", font=("Arial", 14), command=predict)
prediction_button.pack(pady=20)

## Start the GUI application
app.mainloop()