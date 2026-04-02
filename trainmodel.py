import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv('ice-cream.csv')

X = data[['Temperature']]
# Data column is IceCreamsSold in dataset
y = data['IceCreamsSold']

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))