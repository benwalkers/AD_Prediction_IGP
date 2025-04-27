import streamlit as st
import pandas as pd
from stacked_model import StackedXGBFNN

# Load your saved model
@st.cache_resource
def load_model():
    return StackedXGBFNN.load('models')

model = load_model()

# --- App Layout ---
st.title("Alzheimer's Disease Predictor - G15")
st.write("Upload a CSV file with patient data to predict Alzheimer's Disease diagnosis.")

uploaded_file = st.file_uploader("Choose your testing dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data:", test_data.head())

    # Button to trigger prediction
    if st.button("Predict Alzheimer's Diagnosis"):
        try:
            preds = model.predict(test_data)
            proba = model.predict_proba(test_data)
            output = test_data.copy()
            output["Prediction"] = preds
            output["Probability_AD"] = proba
            st.success("Predictions generated successfully!")
            st.write(output)

            # Download option
            csv = output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv,
                file_name="alzheimers_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

