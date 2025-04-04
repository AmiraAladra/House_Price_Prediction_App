import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Page config
st.set_page_config(page_title="🏠 House Price Predictor", page_icon="🏡", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    .stButton > button {
        font-size: 16px;
        padding: 0.5em 2em;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏡 House Price Prediction")
st.write("This app predicts the **selling price** of a house based on key features.")

# Load the model
try:
    with open("models/RFmodel.pkl", "rb") as rf_pickle:
        rf_model = pickle.load(rf_pickle)
    model_loaded = True
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure `models/RFmodel.pkl` exists.")
    model_loaded = False
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model_loaded = False

# Sidebar inputs
st.sidebar.header("🏠 Enter House Details:")
try:
    Num_Beds = st.sidebar.selectbox("Beds", options=["1", "2", "3", "4", "5"])
    Num_Baths = st.sidebar.selectbox("Baths", options=["1", "2", "3", "4", "5", "6"])
    Property_Type = st.sidebar.selectbox("Property Type", options=["Condo", "Bunglow"])
    Basement_Exist = st.sidebar.selectbox("Have Basement", options=["Yes", "No"])
    House_Size = st.sidebar.number_input("Size of the house (sqft)", min_value=0)
    Lot_Size = st.sidebar.number_input("Lot Size", min_value=0)
    Year_Built = st.sidebar.number_input("Year Built", min_value=1880)
    Year_Sold = st.sidebar.number_input("Year Sold", min_value=1880)
    Property_Tax = st.sidebar.number_input("Property Tax", min_value=0)
    Insurance_Amount = st.sidebar.number_input("Insurance", min_value=0)
except Exception as e:
    st.error(f"❌ Error with input form: {e}")

# Center prediction button
col_center = st.columns(3)[1]
with col_center:
    predict_button = st.button("🎯 Predict House Price")

# On button click
if predict_button:
    if not model_loaded:
        st.error("❌ Model is not loaded. Cannot make prediction.")
    else:
        try:
            # Encode inputs
            property_type_Bunglow = 0 if Property_Type == "Condo" else 1
            property_type_Condo = 1 if Property_Type == "Condo" else 0
            basement = 1 if Basement_Exist == "Yes" else 0

            beds = int(Num_Beds)
            baths = int(Num_Baths)
            year_built = int(Year_Built)
            year_sold = int(Year_Sold)
            lot_size = int(Lot_Size)
            sqft = int(House_Size)
            property_tax = int(Property_Tax)
            insurance = int(Insurance_Amount)

            # Derived features
            popular = 1 if (beds == 2 and baths == 2) else 0
            recession = 1 if (2010 <= year_sold <= 2013) else 0
            property_age = year_sold - year_built

            if property_age < 0:
                st.warning("⚠️ 'Year Built' is greater than 'Year Sold'. Please check input values.")
            else:
                prediction_input = [[
                    year_sold, property_tax, insurance, beds, baths, sqft,
                    year_built, lot_size, basement, popular, recession,
                    property_age, property_type_Bunglow, property_type_Condo
                ]]

                new_prediction = rf_model.predict(prediction_input)
                st.success(f"💰 The predicted house price is: **${new_prediction[0]:,.2f}**")

        except ValueError as ve:
            st.error(f"❌ Invalid value in input: {ve}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Visualizations
st.subheader("📊 Data Visualizations")

try:
    col1, col2 = st.columns(2)

    with col1:
        st.image("reports/figures/heatmap.png", caption="Feature Correlation Heatmap", use_column_width=True)

    with col2:
        st.image("reports/figures/boxplot.png", caption="Price by Property Type", use_column_width=True)
except FileNotFoundError:
    st.warning("📉 One or more visualization images are missing.")
except Exception as e:
    st.error(f"❌ Error displaying visualizations: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Random Forest Regression & Streamlit")
