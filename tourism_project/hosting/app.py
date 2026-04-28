import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="suchismitapanda1984/tourism-model",
    filename="model.pkl"
)

model = joblib.load(model_path)

st.title("Tourism Package Prediction")

# User inputs
age = st.number_input("Age", 18, 100)
city = st.selectbox("City Tier", [1, 2, 3])
income = st.number_input("Monthly Income", 1000, 100000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "CityTier": city,
        "MonthlyIncome": income,
        "TypeofContact": "Self Inquiry",
        "Occupation": "Salaried",
        "Gender": "Male",
        "NumberOfPersonVisiting": 1,
        "NumberOfFollowups": 1,
        "ProductPitched": "Basic",
        "PreferredPropertyStar": 3,
        "MaritalStatus": "Single",
        "NumberOfTrips": 1,
        "Passport": 1,
        "PitchSatisfactionScore": 3,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 0,
        "Designation": "Executive",
        "DurationOfPitch": 10
    }])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Customer likely to purchase package")
    else:
        st.error("Customer not likely to purchase")
