import pandas as pd

# Load the dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
df = pd.read_csv(url)

print("Actual column names in dataset:")
print(df.columns.tolist())
print("\n" + "="*60)
print("First 5 rows:")
print(df.head())
print("\n" + "="*60)
print(f"Dataset shape: {df.shape}")
