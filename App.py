import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS for the metric cards
st.markdown("""
    <style>
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("☀️ Solar Power Generation Analytics")
st.markdown("Predict estimated solar energy output based on real-time meteorological conditions.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Sidebar Input Parameters (Updated with your custom strict bounds)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Input Parameters")
    st.markdown("Adjust the environmental factors below:")
    
    with st.expander("🌞 Core Solar & Temp", expanded=True):
        distance_to_solar_noon = st.number_input(
            "Distance to Solar Noon (radians)", 
            min_value=0.0, max_value=1.0, value=0.250000, format="%.6f",
            help="Angle distance to the sun's highest point in radians (0 to 1)."
        )
        temperature = st.slider(
            "Temperature (°C)", 
            min_value=40.0, max_value=80.0, value=60.0, step=0.1,
            help="Daily average temperature in degrees Celsius (40 to 80)."
        )
        sky_cover = st.selectbox(
            "Sky Cover Index", 
            [0, 1, 2, 3, 4], index=0,
            help="0 = Totally clear, 4 = Completely covered overcast (0 to 4)."
        )

    with st.expander("☁️ Atmospheric Data", expanded=False):
        visibility = st.slider(
            "Visibility (km)", 
            min_value=0.0, max_value=10.0, value=10.0, step=0.1,
            help="Visibility distance in kilometers (0 to 10)."
        )
        humidity = st.slider(
            "Humidity (%)", 
            min_value=14, max_value=100, value=75,
            help="Relative humidity percentage (14 to 100)."
        )
        avg_pressure_period = st.slider(
            "Average Pressure (Period)", 
            min_value=29.48, max_value=30.53, value=29.97, step=0.01,
            help="Average barometric pressure in mercury inches (29.48 to 30.53)."
        )

    with st.expander("💨 Wind Conditions", expanded=False):
        wind_direction = st.slider(
            "Wind Direction (1–36)", 
            min_value=1, max_value=36, value=27, step=1,
            help="Daily average wind direction indices (1 to 36)."
        )
        wind_speed = st.slider(
            "Wind Speed (m/s)", 
            min_value=1.0, max_value=27.0, value=8.5, step=0.1,
            help="Daily average wind speed in meters per second (1 to 27)."
        )
        avg_wind_speed_period = st.slider(
            "Average Wind Speed (Period)", 
            min_value=0.0, max_value=40.0, value=5.0, step=0.1,
            help="Average wind speed during the 3-hour period in meters per second (0 to 40)."
        )

    st.write("") # Spacer
    run_prediction = st.button("Calculate Output", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Pipeline Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("Model.pkl")

try:
    artifacts = load_pipeline()
    model = artifacts["model"]
    oe = artifacts["one_hot_encoder"]
    pt = artifacts["power_transformer"]
    scaler = artifacts["standard_scaler"]
    scaler_cols = artifacts["scaler_cols"]
    expected_features = artifacts["expected_features"]
except Exception as e:
    st.sidebar.error("⚠️ Error loading model artifacts. Ensure 'Model.pkl' is present.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Main Dashboard Display & Prediction Logic
# -----------------------------------------------------------------------------
if run_prediction:
    
    # A. Place your UI entries directly into the model layout structure
    input_df = pd.DataFrame([{
        "distance-to-solar-noon": distance_to_solar_noon,
        "temperature": temperature,
        "wind-speed": wind_speed,
        "visibility": visibility,
        "humidity": humidity,
        "average-wind-speed-(period)": avg_wind_speed_period,
        "average-pressure-(period)": avg_pressure_period,
        "wind-direction": wind_direction
    }])
    
    # B. Apply Wind direction transformation exactly matching your notebook logic
    input_df["wind-direction"] = input_df["wind-direction"] * 10
    wind_dir_rad = np.deg2rad(input_df["wind-direction"])
    input_df["wind-dir-sin"] = np.sin(wind_dir_rad)
    input_df["wind-dir-cos"] = np.cos(wind_dir_rad)
    input_df.drop(columns=["wind-direction"], inplace=True)
    
    # C. One-Hot Encoding transformation matching OneHotEncoder setup for 'sky-cover'
    sky_df = pd.DataFrame([[sky_cover]], columns=['sky-cover'])
    encoded_sky = oe.transform(sky_df)
    encoded_sky_cols = pd.DataFrame(encoded_sky, columns=oe.get_feature_names_out(['sky-cover']))
    
    # Combine structural features together
    input_df = pd.concat([input_df, encoded_sky_cols], axis=1)
    
    # D. Correct Skewness via your pre-fitted PowerTransformer object
    input_df[["visibility", "humidity"]] = pt.transform(input_df[["visibility", "humidity"]])
    
    # E. Center and scale data with your pre-fitted StandardScaler object
    input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
    
    # F. Filter down to exact original training sequence structure
    processed_input = input_df[expected_features]
    
    # G. Query LightGBM model for estimate
    prediction = model.predict(processed_input)[0]
    
    # Prevent negative yield values from generating under strange feature choices
    final_power = max(0.0, prediction)

    # Display Metrics Grid Layout
    col_out1, col_out2, col_out3 = st.columns(3)
    with col_out1:
        st.metric(
            label="Estimated Power Generated", 
            value=f"{final_power:,.2f} Joules", 
            help="Predicted energy output in Joules for the 3-hour period."
        )
    with col_out2:
        # Panel temperature efficiency calculation proxy 
        relative_efficiency = max(10, min(100, int(100 - (abs(temperature - 55) * 1.5))))
        st.metric(label="Estimated Panel Efficiency Proxy", value=f"{relative_efficiency}%")
    with col_out3:
        status = "Optimal Yield" if sky_cover <= 1 else ("Moderate Yield" if sky_cover == 2 else "Low Overcast Yield")
        st.metric(label="Condition Profile", value=status)
        
    # Optional Details Section
    with st.expander("🔬 View Processed Input Data Vector", expanded=False):
        st.dataframe(processed_input)

else:
    # Default instruction state before clicking calculate button
    st.info("👈 Click **Calculate Output** on the panel sidebar to see the solar analytics prediction.")