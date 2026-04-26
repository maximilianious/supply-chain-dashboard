import pandas as pd


df = pd.read_csv('data/supply_chain_data.csv')


# Rename columns for easier use (remove spaces)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()


# Drop rows where key columns are missing
df = df.dropna(subset=['revenue_generated', 'number_of_products_sold'])


# Fill remaining missing numbers with the column median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


# Save the cleaned data
df.to_csv('data/clean_data.csv', index=False)
print('Data cleaned! Shape:', df.shape)
