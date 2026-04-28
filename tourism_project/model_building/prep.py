import pandas as pd
import os

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/data.csv")
df = df.drop_duplicates()

df.to_csv("artifacts/cleaned_data.csv", index=False)

print("Preprocessing done")
