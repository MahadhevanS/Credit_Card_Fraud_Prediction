import streamlit as st
import pandas as pd
import joblib

# IMPORTANT: You must have your pre-trained model files in the same directory.
try:
    model = joblib.load("fraud_predictor.pkl")
except FileNotFoundError:
    st.error("Error: fraud_predictor.pkl model file not found.")
    st.stop()

# Define the input parameters and their min/max ranges from your training data
# This dictionary now includes all 28 V-columns with the correct ranges.
parameters = {
    'Time': {'min': 0.0, 'max': 172782.0},
    'V1': {'min': -34.14823365, 'max': 2.43920748},
    'V2': {'min': -48.06085602, 'max': 21.46720299},
    'V3': {'min': -33.68098402, 'max': 9.38255843},
    'V4': {'min': -5.56011758, 'max': 12.69954198},
    'V5': {'min': -23.66972569, 'max': 29.01612354},
    'V6': {'min': -20.86962619, 'max': 16.4932271},
    'V7': {'min': -41.50679608, 'max': 21.43751446},
    'V8': {'min': -50.42009006, 'max': 19.16832739},
    'V9': {'min': -13.43406632, 'max': 15.59499461},
    'V10': {'min': -24.40318497, 'max': 23.74513612},
    'V11': {'min': -4.68293055, 'max': 11.61972348},
    'V12': {'min': -18.43113103, 'max': 4.84645241},
    'V13': {'min': -4.00863979, 'max': 4.0993519},
    'V14': {'min': -18.82208674, 'max': 7.75459875},
    'V15': {'min': -4.49894468, 'max': 4.1985829},
    'V16': {'min': -13.25154198, 'max': 4.7343215},
    'V17': {'min': -22.88399858, 'max': 7.73345628},
    'V18': {'min': -9.28783221, 'max': 4.09343996},
    'V19': {'min': -4.93273306, 'max': 4.47512691},
    'V20': {'min': -21.53382174, 'max': 14.93500045},
    'V21': {'min': -22.88934704, 'max': 27.20283916},
    'V22': {'min': -8.88701714, 'max': 8.36198519},
    'V23': {'min': -22.57500044, 'max': 22.08354487},
    'V24': {'min': -2.8248489, 'max': 3.99064595},
    'V25': {'min': -3.96345399, 'max': 6.07085038},
    'V26': {'min': -2.06856087, 'max': 3.00445539},
    'V27': {'min': -22.56567932, 'max': 9.20088257},
    'V28': {'min': -11.71089564, 'max': 15.94215098},
    'Amount': {'min': 0.0, 'max': 5239.5}
}


def fraud_predictor_app():
    st.title("💳 Transaction Fraud Detector")
    st.write("Enter the transaction details to predict if it is fraudulent.")
    
    # Store inputs in a dictionary
    input_data = {}
    
    # Create inputs for all required features
    # Use st.columns to manage layout
    cols = st.columns(2)
    
    # Create inputs for Time and Amount in the first column
    with cols[0]:
        st.subheader("Transaction Details")
        for feature in ['Time', 'Amount']:
            input_data[feature] = st.number_input(
                feature,
                min_value=parameters[feature]['min'],
                max_value=parameters[feature]['max'],
                help=f"Range: {parameters[feature]['min']} to {parameters[feature]['max']}"
            )
            
    # Create inputs for V-columns in the second column
    with cols[1]:
        st.subheader("Dimensionality Reduction Features (V-columns)")
        for feature in parameters:
            # Skip Time and Amount as they are already handled
            if feature in ['Time', 'Amount']:
                continue
            
            input_data[feature] = st.number_input(
                feature,
                min_value=parameters[feature]['min'],
                max_value=parameters[feature]['max'],
                format="%.8f",
                help=f"Range: {parameters[feature]['min']:.8f} to {parameters[feature]['max']:.8f}"
            )
    
    st.markdown("---")
    
    # Predict button
    if st.button("Predict"):
        # Create a DataFrame from the collected inputs
        input_df = pd.DataFrame([input_data])
        
        # Ensure the columns are in the correct order for the model
        input_df = input_df[parameters.keys()]
        
        try:
            # Predict using the model's .predict() method
            prediction = model.predict(input_df)
            
            # Use the .decision_function() to get the anomaly score
            anomaly_score = model.decision_function(input_df)[0]
            
            # IsolationForest returns -1 for outliers/anomalies
            if prediction[0] == -1:
                st.error("🚨 Warning: This transaction is predicted to be **FRAUDULENT**.")
                st.write(f"Anomaly Score: {anomaly_score:.2f} (More negative indicates higher likelihood of fraud)")
            else:
                st.success("✅ This transaction is predicted to be **SAFE**.")
                st.write(f"Anomaly Score: {anomaly_score:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# This part allows the app to be run as a standalone file or called from your main hub
if __name__ == "__main__":
    fraud_predictor_app()
