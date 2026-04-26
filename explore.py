import pandas as pd


# Load the dataset
df = pd.read_csv('data/supply_chain_data.csv')


# Basic info
print('Shape:', df.shape)          # How many rows and columns
print('\nColumns:', df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())


# Check for missing values
print('\nMissing values per column:')
print(df.isnull().sum())


# Basic statistics
print('\nBasic statistics:')
print(df.describe())
