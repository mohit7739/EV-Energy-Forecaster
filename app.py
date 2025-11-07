import streamlit as st

# Set the title of the web app
st.title("My EV Smart Range Forecaster 🚗⚡")

# Write a welcome message
st.write("Welcome to the first version of my project!")
st.write("This app will use a machine learning model to predict EV energy consumption.")

# Show a simple header
st.header("Model Evaluation Results")

# Display the great results you just got!
st.write("I've successfully trained a model in Week 2. Here are the results:")
st.metric(label="R-squared (R²)", value="94.61 %")
st.metric(label="Mean Absolute Error (MAE)", value="0.4106 kWh")

st.write("Next steps: Integrate the model and build a Gen AI chatbot!")