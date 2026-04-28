import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("artifacts/cleaned_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("artifacts/model.pkl")

preds = model.predict(X)

print("Accuracy:", accuracy_score(y, preds))
