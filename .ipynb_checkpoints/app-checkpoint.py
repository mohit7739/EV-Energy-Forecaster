import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import google.generativeai.protos as protos # For older library compatibility

# --- Page Setup ---
st.set_page_config(page_title="EV Smart Range Forecaster", page_icon="🚗")
st.title("🚗 EV Smart Range Forecaster")
st.subheader("Predicting Energy Consumption (kWh) in Real-Time")

# --- Cached Functions to Load Models and API ---

@st.cache_data
def load_models():
    """Loads the saved ML model and column list from disk."""
    try:
        model = joblib.load('model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found! Please run the notebook to generate 'model.pkl' and 'model_columns.pkl'.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None, None

@st.cache_resource
def load_genai_model():
    """Connects to the Gemini API and returns the model."""
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=API_KEY)
        
        # Create the model WITHOUT the tool config
        genai_model = genai.GenerativeModel(
            model_name="models/gemini-pro-latest",
            generation_config=GenerationConfig(temperature=0.7) # Temp 0.7 for creative chat
        )
        return genai_model
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {e}. Check your API key in .streamlit/secrets.toml")
        return None

# --- Local Python Function to run the ML model ---
def predict_energy(model, model_columns, 
                   speed_kmh=60.0, temperature_c=20.0, slope_percent=0.0, 
                   driving_mode=1, road_type=1, traffic_condition=1, weather_condition=1,
                   acceleration_ms2=0.0, battery_state_percent=80.0, battery_voltage_v=350.0,
                   battery_temp_c=25.0, humidity_percent=50.0, wind_speed_ms=5.0,
                   tire_pressure_psi=32.0, vehicle_weight_kg=1800.0, distance_travelled_km=10.0):
    """
    This is the actual Python function that runs your saved scikit-learn model.
    """
    input_data = {
        'Speed_kmh': speed_kmh,
        'Acceleration_ms2': acceleration_ms2,
        'Battery_State_%': battery_state_percent,
        'Battery_Voltage_V': battery_voltage_v,
        'Battery_Temperature_C': battery_temp_c,
        'Slope_%': slope_percent,
        'Temperature_C': temperature_c,
        'Humidity_%': humidity_percent,
        'Wind_Speed_ms': wind_speed_ms,
        'Tire_Pressure_psi': tire_pressure_psi,
        'Vehicle_Weight_kg': vehicle_weight_kg,
        'Distance_Travelled_km': distance_travelled_km,
        'Driving_Mode': driving_mode,
        'Road_Type': road_type,
        'Traffic_Condition': traffic_condition,
        'Weather_Condition': weather_condition
    }
    
    input_df = pd.DataFrame([input_data])
    categorical_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df_aligned)
    return prediction[0]


# --- Load Models and API ---
model, model_columns = load_models()
genai_model = load_genai_model()

if not model or not model_columns: st.stop()
else: st.success("Machine Learning model and columns loaded successfully!")

if not genai_model: st.stop()
else: st.success("Gemini AI Chatbot is connected!")

# Start a simple chat session (no tools)
@st.cache_resource
def start_chat_session(_genai_model):
    return _genai_model.start_chat()

chat = start_chat_session(genai_model)


# --- User Input Sidebar ---
st.sidebar.header("Enter Data for ML Prediction")
speed_kmh_slider = st.sidebar.slider("Speed (km/h)", min_value=0.0, max_value=120.0, value=60.0, step=0.1)
temperature_c_slider = st.sidebar.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=20.0, step=0.1)
slope_percent_slider = st.sidebar.slider("Road Slope (%)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
acceleration_ms2_slider = st.sidebar.slider("Acceleration (m/s²)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
battery_state_percent_slider = st.sidebar.slider("Battery State (%)", min_value=20.0, max_value=100.0, value=80.0)
battery_voltage_v_slider = st.sidebar.slider("Battery Voltage (V)", min_value=300.0, max_value=400.0, value=350.0)
battery_temp_c_slider = st.sidebar.slider("Battery Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0)
humidity_percent_slider = st.sidebar.slider("Humidity (%)", min_value=20.0, max_value=90.0, value=50.0)
wind_speed_ms_slider = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=5.0)
tire_pressure_psi_slider = st.sidebar.slider("Tire Pressure (psi)", min_value=28.0, max_value=35.0, value=32.0)
vehicle_weight_kg_slider = st.sidebar.slider("Vehicle Weight (kg)", min_value=1200.0, max_value=2500.0, value=1800.0)
distance_travelled_km_slider = st.sidebar.slider("Distance Travelled (km)", min_value=0.0, max_value=50.0, value=10.0)
driving_mode_slider = st.sidebar.selectbox("Driving Mode", [1, 2, 3], format_func=lambda x: {1: 'Eco', 2: 'Normal', 3: 'Sport'}.get(x, 'Unknown'))
road_type_slider = st.sidebar.selectbox("Road Type", [1, 2, 3], format_func=lambda x: {1: 'Highway', 2: 'Urban', 3: 'Rural'}.get(x, 'Unknown'))
traffic_condition_slider = st.sidebar.selectbox("Traffic Condition", [1, 2, 3], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High'}.get(x, 'Unknown'))
weather_condition_slider = st.sidebar.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: {1: 'Sunny', 2: 'Rainy', 3: 'Cloudy', 4: 'Snowy'}.get(x, 'Unknown'))


# --- ML Prediction Logic (Manual) ---
if st.sidebar.button("Predict Energy Consumption (Manual)"):
    prediction = predict_energy(
        model=model, model_columns=model_columns,
        speed_kmh=speed_kmh_slider, temperature_c=temperature_c_slider, 
        slope_percent=slope_percent_slider, driving_mode=driving_mode_slider, 
        road_type=road_type_slider, traffic_condition=traffic_condition_slider, 
        weather_condition=weather_condition_slider,
        acceleration_ms2=acceleration_ms2_slider, battery_state_percent=battery_state_percent_slider,
        battery_voltage_v=battery_voltage_v_slider, battery_temp_c=battery_temp_c_slider,
        humidity_percent=humidity_percent_slider, wind_speed_ms=wind_speed_ms_slider,
        tire_pressure_psi=tire_pressure_psi_slider, vehicle_weight_kg=vehicle_weight_kg_slider,
        distance_travelled_km=distance_travelled_km_slider
    )
    
    st.header("⚡ Manual Prediction Result")
    st.metric(label="Predicted Energy Consumption", value=f"{prediction:.2f} kWh")

# --- DIVIDER ---
st.divider()

# --- GEN AI CHATBOT (Simple Version) ---
st.header("🤖 EV Chat Assistant")
st.write("You can ask general questions about EVs, or ask for help interpreting your prediction result.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("Ask a general question about EVs..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- This is now only ONE API call ---
    try:
        response = chat.send_message(prompt)
        ai_response = response.text
        
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
    except Exception as e:
        st.error(f"An error occurred: {e}")