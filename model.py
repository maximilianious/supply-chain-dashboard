import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


df = pd.read_csv('data/clean_data.csv')


# Select features (inputs) and target (what we predict)
features = ['price', 'availability', 'lead_times', 'shipping_times',
            'shipping_costs', 'stock_levels']
target = 'number_of_products_sold'


# Drop rows where any feature or target is missing
model_df = df[features + [target]].dropna()


X = model_df[features]
y = model_df[target]


# Split: 80% to train the model, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared Score: {r2:.4f}')


# Save the model so Streamlit can load it
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


print('Model saved to model.pkl')
