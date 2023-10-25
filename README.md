# Housing Price Prediction in King County

This repository contains the code and resources used to build a Housing Price Prediction model for properties in King County. The model predicts the price of houses based on various features such as number of bedrooms, bathrooms, living area square footage, lot area square footage, condition, grade, and more.

---

**Streamlit App Link: [House Price Prediction App](https://houseprediction.streamlit.app/)**

---

## Overview

The Housing Price Prediction model was developed using Python and popular machine learning libraries. This README provides an overview of the steps taken to create the model and deploy it as a Streamlit web application.

### Data Collection

The dataset used for this project contains a variety of features including the number of bedrooms, bathrooms, square footage of living area, lot area, condition, grade, and many others. The dataset was collected from various sources and served as the foundation for training and evaluating the model.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to gain insights into the dataset. This involved:

- Visualizing data distributions.
- Identifying missing values.
- Analyzing correlations between features.
- Exploring the relationships between price and other features.

### Preprocessing

Data preprocessing was performed to ensure the data was ready for machine learning. This included:

- Handling outliers in the data.
- Preparing the data for model input.

### Model Selection and Hyperparameter Tuning
XGBoost's XGBRegressor was the chosen model due to its impressive performance on structured data. Optuna was used to conduct hyperparameter tuning, in which the model's settings were tweaked to minimize mean squared error, thus improving the accuracy of the model to achieve an R2 score of 0.92.

### Streamlit App Deployment

The selected model was deployed as a Streamlit web application. Users can input various features of a property, and the app provides a real-time prediction of the house price. The app's user interface was designed to be intuitive and user-friendly.


**Note**: This project is for educational purposes and should not be considered professional financial or real estate advice.