from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("data/raw/nasa93.arff")
df = pd.DataFrame(data)

# text columns come as bytes (b'...'), convert them to normal strings
for col in df.select_dtypes(include=object).columns:
    df[col] = df[col].str.decode("utf-8")

print("Shape (rows, columns):", df.shape)
print("\nColumns:\n", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

missing = df.isna().sum()
print("\nMissing values:\n", missing[missing > 0] if missing.any() else "none")

print("\nLast column (should be actual effort):", df.columns[-1])
print(df.iloc[:, -1].describe())