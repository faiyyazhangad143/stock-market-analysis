import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Load the dataset
data = pd.read_csv("500325.csv")

# Data processing
data['Date'] = pd.to_datetime(data['Date'], format='%d-%B-%Y')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Year', 'Month', 'Day']], data['Close Price'], test_size=0.2)

# Train the model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('Reliance Share Price Prediction')

# Get user input using a date picker
selected_date = st.date_input('Select a date', min_value=pd.to_datetime('2024-01-01'), max_value=pd.to_datetime('2029-12-31'))

# Extract year, month, and day from the selected date
year = selected_date.year
month = selected_date.month
day = selected_date.day

# Predict the share price
if st.button('Predict'):
    future_data = np.array([[year, month, day]])
    predicted_price = model.predict(future_data)[0]
    st.write('Predicted share price:', predicted_price)
