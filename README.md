🚗 EV Smart Range Forecaster

This is a web application that predicts the energy consumption (kWh) of an Electric Vehicle using a machine learning model. It also features a Generative AI chatbot to answer questions about EVs.

This project was built in Streamlit for an internship, demonstrating skills in data science, machine learning, and AI integration.

🌟 Features

Manual ML Prediction: Use the sliders and dropdowns in the "Manual Prediction Controls" to set real-world conditions (speed, temperature, weather, etc.) and get a highly accurate prediction from a trained scikit-learn model.

AI Chat Assistant: Ask the chatbot general questions about electric vehicles (e.g., "What is regenerative braking?").

"Smart" Chat Predictions (Hybrid Model): Ask the chatbot for a prediction in plain English!

e.g., "What's my energy consumption at 90 km/h in 30C weather on a highway in the rain?"

The chatbot uses Function Calling to extract the data, runs the local ML model, and provides the prediction, all while using only one API call to stay within free-tier limits.

🛠️ How it Works

ML Model: A Multi-Variate Linear Regression model was trained in a Jupyter Notebook (EV-Energy-Forecaster.ipynb) on the EV Energy Consumption Dataset.

The model achieved an R-squared (R²) of 94.61%.

The model (model.pkl) and its column order (model_columns.pkl) are saved locally.

Streamlit App (app.py):

The app loads the saved model.pkl file for manual predictions.

It connects to the Google Gemini API using a secret API key (.streamlit/secrets.toml).

It uses Streamlit's chat elements to create a user-friendly interface.

Hybrid Chat (Function Calling):

The Gemini AI model is configured with a "tool" (a function definition) that matches our local predict_energy function.

When a user asks for a prediction, the AI's "Tool Use" mode is triggered.

The AI's only response is a function_call object, which the app intercepts (this is API Call #1).

The app uses the arguments from the function_call to run the local model.pkl file.

The app formats the numerical result into a friendly string and displays it, skipping the second API call to avoid hitting per-minute rate limits.

🏃‍♂️ How to Run

Clone the Repository:

git clone [your-repo-url]
cd [your-repo-name]


Create and activate a virtual environment:

python3 -m venv env
source env/bin/activate


Install Dependencies:

pip install streamlit pandas joblib scikit-learn google-generativeai


Create Your Secret API Key:

Create a folder: mkdir .streamlit

Create a secrets file: nano .streamlit/secrets.toml

Add your Gemini API key to this file:

GEMINI_API_KEY = "YOUR_API_KEY_HERE"


Save and exit (Ctrl+O, Enter, Ctrl+X).

Run the App:

streamlit run app.py
