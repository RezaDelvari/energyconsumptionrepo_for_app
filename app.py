import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the model
#with open('lgbm_energy_prediction1.pkl', 'rb') as file:
    #lgbm_model = pickle.load(file)
# Load the model back
lgbm_model = joblib.load('lgbm_energy_prediction2.pkl')
# Function to make predictions
def make_prediction(input_data):
    # Preprocess input data if necessary
    # Make prediction using the loaded model
    prediction = lgbm_model.predict(input_data)
    return prediction
# Streamlit app
def main():
    # Custom CSS styling to adjust the main content area
    st.markdown(
        """
        <style>
        .main-container {
            margin-left: 200px; /* Make space for the sidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # LinkedIn barcode image in the sidebar
    st.sidebar.title('LinkedIn Barcode')
    st.sidebar.image('1.jpg', use_column_width=False, output_format='JPEG', width=300)

    # Add "Created by Reza" with a hyperlink to your GitHub page at the bottom of the sidebar
    st.sidebar.markdown(
        """  
        ---
        Created by [Reza](https://github.com/RezaDelvari)
        """
    )

    # Main content area
    st.title('Energy Prediction App')
    st.write('Enter the required information to predict energy consumption')

    # Input widgets
    is_business = st.checkbox('Is Business?')
    is_consumption = st.checkbox('Is Consumption?')
    year = st.number_input('Year', min_value=2023, max_value=2100, value=2023, step=1)
    month = st.selectbox('Month', range(1, 13), index=0)
    day = st.selectbox('Day', range(1, 32), index=0)
    hour = st.slider('Hour', 0, 23, 0)

    # Set installed_capacity to 0 if is_consumption is checked
    installed_capacity = 0 if is_consumption else st.number_input('Installed Capacity', value=1438.93017)

    # Default values based on provided statistics
    default_total_precipitation_mean_f = 0.00008
    default_cloudcover_low_min_f = 0.28421
    default_direct_solar_radiation_mean_f = 146.44857

    total_precipitation_mean_f = st.number_input('Total Precipitation Mean (F)', value=default_total_precipitation_mean_f)
    cloudcover_low_min_f = st.number_input('Cloudcover Low Min (F)', value=default_cloudcover_low_min_f)
    direct_solar_radiation_mean_f = st.number_input('Direct Solar Radiation Mean (F)', value=default_direct_solar_radiation_mean_f)

    # Make prediction when a button is clicked
    if st.button('Predict'):
        input_data = [[is_business, is_consumption, year, month, day, hour, installed_capacity, total_precipitation_mean_f, cloudcover_low_min_f, direct_solar_radiation_mean_f]]
        prediction = make_prediction(input_data)
        if is_consumption:
            st.write('Predicted energy consumption:', f'{prediction[0]:,.2f} kWh')  # Display result with kWh unit
        else:
            st.write('Predicted energy production:', f'{prediction[0]:,.2f} kWh')  # Display result with kWh unit

if __name__ == '__main__':
    main()





