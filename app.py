from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "california_housing_model.pkl"


st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the notebook first to create "
            "models/california_housing_model.pkl."
        )
    return joblib.load(MODEL_PATH)


st.title("California Housing Price Prediction")
st.write(
    "Enter the housing and location information to estimate "
    "the median house value using the trained KNN regression pipeline."
)

try:
    model = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.header("Housing Information")

with st.form("housing_prediction_form"):
    medinc = st.number_input("Median Income", min_value=0.0, value=3.0, step=0.1)
    houseage = st.number_input("House Age", min_value=0.0, value=20.0, step=1.0)
    averooms = st.number_input("Average Rooms", min_value=0.0, value=5.0, step=0.1)
    avebedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0, step=0.1)
    population = st.number_input("Population", min_value=0.0, value=1000.0, step=10.0)
    aveoccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.0, step=0.1)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)

    submitted = st.form_submit_button("Predict House Price", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(input_data)[0]
    predicted_price = prediction * 100000

    st.success(f"Predicted Median House Value: ${predicted_price:,.2f}")
    st.caption(
        "This is an estimate based on historical California census data, "
        "not a formal property valuation."
    )
