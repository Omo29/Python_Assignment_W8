import pandas as pd

# 1. Load the data
df = pd.read_csv('metadata.csv')

# 2. Examine first rows and structure
print("First 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Basic statistics for numerical columns
print("\nDescriptive statistics:")
print(df.describe(include='number'))   # only numeric types
