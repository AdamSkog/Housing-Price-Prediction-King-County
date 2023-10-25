import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the model
model = joblib.load("app/models/xgbmodel.pkl")
data = pd.read_csv("data/raw/kc_house_data.csv")


# Function to make predictions
def predict_price(
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    grade,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    zipcode,
    lat,
    long,
    sqft_living15,
    sqft_lot15,
):
    input_data = pd.DataFrame(
        {
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "grade": [grade],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "zipcode": [zipcode],
            "lat": [lat],
            "long": [long],
            "sqft_living15": [sqft_living15],
            "sqft_lot15": [sqft_lot15],
        },
        index=[0],
    )
    prediction = model.predict(input_data)
    return prediction[0]


# Streamlit UI
st.title("King County House Price Prediction 🏠")
st.sidebar.header("Input Features")

# Input fields
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=10.0, step=0.5)
sqft_living = st.sidebar.slider(
    "Living Area (sqft)", min_value=500, max_value=10000, step=100
)
sqft_lot = st.sidebar.slider(
    "Lot Area (sqft)", min_value=500, max_value=100000, step=500
)
floors = st.sidebar.slider("Floors", min_value=1, max_value=3, step=1)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.slider("View", min_value=0, max_value=4, step=1)
condition = st.sidebar.slider("Condition", min_value=1, max_value=5, step=1)
grade = st.sidebar.slider("Grade", min_value=1, max_value=13, step=1)
sqft_above = st.sidebar.slider(
    "Sqft Above Ground", min_value=500, max_value=8000, step=100
)
sqft_basement = st.sidebar.slider(
    "Sqft Basement", min_value=0, max_value=5000, step=100
)
yr_built = st.sidebar.slider("Year Built", min_value=1900, max_value=2022, step=1)
yr_renovated = st.sidebar.slider(
    "Year Renovated", min_value=1900, max_value=2022, step=1
)
zipcode = st.sidebar.number_input("Zipcode", min_value=98001, max_value=98199, step=1)
lat = st.sidebar.slider("Latitude", min_value=47.0, max_value=48.0, step=0.01)
long = st.sidebar.slider("Longitude", min_value=-123.0, max_value=-122.0, step=0.01)
sqft_living15 = st.sidebar.slider(
    "Living Area of Neighbors (sqft)", min_value=500, max_value=10000, step=100
)
sqft_lot15 = st.sidebar.slider(
    "Lot Area of Neighbors (sqft)", min_value=500, max_value=100000, step=500
)

# Predict and display the result
if st.sidebar.button("Predict"):
    prediction = predict_price(
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        grade,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated,
        zipcode,
        lat,
        long,
        sqft_living15,
        sqft_lot15,
    )
    print(prediction)
    st.success(f"Predicted Price: ${prediction:.2f}")

st.write(
    """
This app predicts the price of houses based on various features. Please adjust the input parameters to get the prediction!
"""
)

# Input section
st.header("Input Features")

feature_explanations = """
price - Price of each home sold\n
bedrooms - Number of bedrooms\n
bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower\n
sqft_living - Square footage of the apartments interior living space\n
sqft_lot - Square footage of the land space\n
floors - Number of floors\n
waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not\n
view - An index from 0 to 4 of how good the view of the property was\n
condition - An index from 1 to 5 on the condition of the apartment,\n
grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.\n
sqft_above - The square footage of the interior housing space that is above ground level\n
sqft_basement - The square footage of the interior housing space that is below ground level\n
yr_built - The year the house was initially built\n
yr_renovated - The year of the house’s last renovation\n
zipcode - What zipcode area the house is in\n
lat - Lattitude\n
long - Longitude\n
sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors\n
sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors\n
"""

st.write(feature_explanations)

# Visualization section
st.header("Data Visualization")

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig = plt.figure(figsize=(12, 10))
sns.heatmap(
    data.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm", center=0
)
st.pyplot(fig)

# Distribution of target variable
st.subheader("Distribution of Target Variable (Price)")
fig = plt.figure(figsize=(8, 6))
sns.histplot(data["price"], bins=30, kde=True)
plt.xlabel("Price")
plt.ylabel("Frequency")
st.pyplot(fig)
