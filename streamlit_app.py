import streamlit as st
import pandas as pd
import shap
import numpy as np
import plotly.graph_objects as go
from stacked_model import StackedXGBFNN

# Load your saved model
# @st.cache_resource
# def load_model():
#     return StackedXGBFNN.load('models')

# model = load_model()

# # --- App Layout ---
# st.title("Alzheimer's Disease Predictor - G15")
# st.write("Upload a CSV file with patient data to predict Alzheimer's Disease diagnosis.")

# uploaded_file = st.file_uploader("Choose your testing dataset (CSV format)", type=["csv"])

# if uploaded_file is not None:
#     test_data = pd.read_csv(uploaded_file)
#     st.write("### Preview of uploaded data:", test_data.head())

#     # Button to trigger prediction
#     if st.button("Predict Alzheimer's Diagnosis"):
#         try:
#             preds = model.predict(test_data)
#             proba = model.predict_proba(test_data)
#             output = test_data.copy()
#             output["Prediction"] = preds
#             output["Probability_AD"] = proba
#             st.success("Predictions generated successfully!")
#             st.write(output)

#             # Download option
#             csv = output.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="📥 Download predictions as CSV",
#                 data=csv,
#                 file_name="alzheimers_predictions.csv",
#                 mime="text/csv"
#             )
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
# Load your saved stacked model
@st.cache_resource
def load_model():
    return StackedXGBFNN.load('models')

model = load_model()

# --- App Layout ---
st.set_page_config(page_title="Alzheimer's Disease Predictor - G15", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 Alzheimer's Disease Predictor - G15</h1>", unsafe_allow_html=True)
st.write("Upload a CSV file with patient data to predict Alzheimer's Disease diagnosis.")

uploaded_file = st.file_uploader("Choose your testing dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_data.head())

    # Predict button
    if st.button("Predict Alzheimer's Diagnosis"):
        try:
            preds = model.predict(test_data)
            probas = model.predict_proba(test_data)

            # Prepare output DataFrame
            output = test_data.copy()
            output['Prediction'] = ['AD' if p == 1 else 'Non-AD' for p in preds]
            output['Probability_AD'] = probas

            st.success("✅ Predictions generated successfully!")
            st.dataframe(output)

            # Download results
            csv = output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv,
                file_name="alzheimers_predictions.csv",
                mime="text/csv"
            )

            # SHAP feature contribution
            st.markdown("---")
            st.write("### SHAP Feature Contribution")
            record_idx = st.number_input(
                "Select record index to explain:",
                min_value=0,
                max_value=len(test_data) - 1,
                value=0,
                step=1
            )
            show_shap = st.checkbox("Show SHAP plot for selected record")

            if show_shap:
                # Preprocess selected record
                selected_record = test_data.iloc[[record_idx]]
                X_proc = model.preprocess_pipe.transform(selected_record)

                # Explain with SHAP (using XGBoost part)
                explainer = shap.Explainer(model.xgb_model)
                shap_vals = explainer(X_proc)

                # Extract SHAP values and feature names
                vals = shap_vals[0].values
                feat_names = model.preprocess_pipe.get_feature_names_out()
                idx_sorted = np.argsort(np.abs(vals))[::-1]
                sorted_feats = feat_names[idx_sorted]
                sorted_vals = vals[idx_sorted]

                # Build Plotly bar chart
                fig = go.Figure(go.Bar(
                    x=sorted_vals,
                    y=sorted_feats,
                    orientation='h',
                    marker_color=['green' if v > 0 else 'red' for v in sorted_vals]
                ))
                fig.update_layout(
                    title=f"SHAP Contributions for Record {record_idx}",
                    xaxis_title="SHAP Value (Impact on Log-Odds)",
                    yaxis_title="Feature",
                    template="plotly_white",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
