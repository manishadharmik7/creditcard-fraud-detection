import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

# ✅ Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=28,
    n_informative=10,
    n_redundant=5,
    weights=[0.97, 0.03],
    random_state=42
)

# Match feature structure your model expects
df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df["Time"] = np.random.uniform(0, 100000, len(df))
df["Amount"] = np.random.uniform(0, 2000, len(df))
df["Hour"] = (df["Time"] / 3600) % 24
df["Log_Amount"] = np.log1p(df["Amount"])
df["Class"] = y

# Save
df = df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour", "Log_Amount", "Class"]]
df.to_csv("data/creditcard.csv", index=False)

print("✅ creditcard.csv generated successfully.")
print(df.head())
print(df['Class'].value_counts())
