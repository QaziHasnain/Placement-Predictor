import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Initial look at the data
print(df.head())
print(df.info()) # Check for data types and non-null counts
