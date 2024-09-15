import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pickled XGBoost model and feature names
with open('xgboost_diagnosis_model.pkl', 'rb') as file:
    model, feature_names = pickle.load(file)

# Define the input features for the model
def user_input_features():
    st.sidebar.header('Enter Patient Information')
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=10, step=1)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=50.0, value=18.5, step=0.1)
    alvarado_score = st.sidebar.number_input('Alvarado Score', min_value=0, max_value=10, value=5, step=1)
    body_temperature = st.sidebar.number_input('Body Temperature (°C)', min_value=35.0, max_value=42.0, value=37.0, step=0.1)
    wbc_count = st.sidebar.number_input('WBC Count', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    crp = st.sidebar.number_input('CRP', min_value=0.0, max_value=200.0, value=5.0, step=0.1)

    # Example categorical inputs encoded as binary (e.g., yes = 1, no = 0)
    sex_male = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    appendix_on_us = st.sidebar.selectbox('Appendix on Ultrasound (US)', ('Yes', 'No'))
    lower_right_abd_pain = st.sidebar.selectbox('Lower Right Abdomen Pain', ('Yes', 'No'))
    migratory_pain = st.sidebar.selectbox('Migratory Pain', ('Yes', 'No'))
    neutrophilia = st.sidebar.selectbox('Neutrophilia', ('Yes', 'No'))

    # Encode categorical features
    sex_male_encoded = 1 if sex_male == 'Male' else 0
    appendix_on_us_encoded = 1 if appendix_on_us == 'Yes' else 0
    lower_right_abd_pain_encoded = 1 if lower_right_abd_pain == 'Yes' else 0
    migratory_pain_encoded = 1 if migratory_pain == 'Yes' else 0
    neutrophilia_encoded = 1 if neutrophilia == 'Yes' else 0

    # Create a dictionary of input features
    features_dict = {
        'Age': age,
        'BMI': bmi,
        'Alvarado_Score': alvarado_score,
        'Body_Temperature': body_temperature,
        'WBC_Count': wbc_count,
        'CRP': crp,
        'Sex_male': sex_male_encoded,
        'Appendix_on_US_yes': appendix_on_us_encoded,
        'Lower_Right_Abd_Pain_yes': lower_right_abd_pain_encoded,
        'Migratory_Pain_yes': migratory_pain_encoded,
        'Neutrophilia_yes': neutrophilia_encoded
    }

    return features_dict

def main():
    st.title('Pediatric Appendicitis Diagnosis Prediction')
    st.write("""
    This application predicts whether a pediatric patient has appendicitis based on input clinical features.
    """)

    # Get user input features
    input_features_dict = user_input_features()

    # Create a DataFrame with the user's input
    input_features_df = pd.DataFrame([input_features_dict])

    # Ensure all 59 features are present by reindexing the DataFrame
    input_features_df = input_features_df.reindex(columns=feature_names, fill_value=0)

    # Add a submit button
    if st.button('Submit'):
        # Make predictions
        prediction = model.predict(input_features_df)
        prediction_proba = model.predict_proba(input_features_df)[:, 1]  # Probability for the positive class (appendicitis)

        # Display the result in a medical interpretive way
        st.subheader('Prediction Interpretation')
        probability = prediction_proba[0] * 100  # Convert to percentage
        if prediction[0] == 1:
            diagnosis = f'The model predicts a {probability:.2f}% probability that the patient has appendicitis.'
        else:
            diagnosis = f'The model predicts a {100 - probability:.2f}% probability that the patient does not have appendicitis.'
        
        st.write(diagnosis)

        # Provide some context to the probability
        st.subheader('Medical Context')
        if probability > 75:
            st.write('High probability of appendicitis. Recommend further clinical evaluation and potential surgical consultation.')
        elif 50 < probability <= 75:
            st.write('Moderate probability of appendicitis. Suggest monitoring and further diagnostic tests (e.g., imaging, lab tests).')
        else:
            st.write('Low probability of appendicitis. Observation and conservative management may be sufficient, but clinical judgment is crucial.')

if __name__ == '__main__':
    main()