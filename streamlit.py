import pickle
import numpy as np

import streamlit as st

# Load the model and scaler
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))

# Function to make predictions
def predict_price(data):
  # Transform the data
  new_data = scalar.transform(np.array(data).reshape(1, -1))
  # Make prediction
  prediction = regmodel.predict(new_data)[0]
  return prediction

# Streamlit App
st.set_page_config(page_title="California House Price Prediction")

# Title
st.title("California House Price Prediction")

# Input fields with clear labels
medinc = st.text_input("Median Income", "", placeholder="Enter median income")
house_age = st.text_input("House Age", "", placeholder="Enter house age")
ave_rooms = st.text_input("Average Rooms", "", placeholder="Enter average number of rooms")
ave_bedrms = st.text_input("Average Bedrooms", "", placeholder="Enter average number of bedrooms")
population = st.text_input("Population", "", placeholder="Enter population")
ave_occup = st.text_input("Average Occupancy", "", placeholder="Enter average occupancy")
latitude = st.text_input("Latitude", "", placeholder="Enter latitude")
longitude = st.text_input("Longitude", "", placeholder="Enter longitude")

# Submit button
if st.button("Predict"):
  # Error handling for invalid input
  try:
    data = [float(x) for x in [medinc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]]
    # Make prediction
    prediction = predict_price(data)
    prediction_text = f"Predicted House Price: ${prediction:.2f}"
    st.write(prediction_text)
  except ValueError:
    st.error("Please enter valid numerical features for all fields.")

