'''
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Year', 'Month']], data['Close Price'], test_size=0.2)

# Train the model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('Reliance Share Price Prediction')

# Get user input
month = st.number_input('Enter the month (1-12)', min_value=1, max_value=12)
year = st.number_input('Enter the year (2024-2029)', min_value=2024, max_value=2029)

# Predict the share price
if st.button('Predict'):
    future_data = np.array([[year, month]])
    predicted_price = model.predict(future_data)[0]
    st.write('Predicted share price:', predicted_price)
'''



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

# Get user input
day = st.number_input('Enter the day (1-31)', min_value=1, max_value=31)
month = st.number_input('Enter the month (1-12)', min_value=1, max_value=12)
year = st.number_input('Enter the year (2024-2029)', min_value=2024, max_value=2029)

# Predict the share price
if st.button('Predict'):
    future_data = np.array([[year, month, day]])
    predicted_price = model.predict(future_data)[0]
    st.write('Predicted share price:', predicted_price)
