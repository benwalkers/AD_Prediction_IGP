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

# --------------------------------------------------
# Page configuration (must be first Streamlit call)
# --------------------------------------------------
st.set_page_config(page_title="Alzheimer's Disease Predictor - G15", layout="wide")

# --------------------------------------------------
# Title and intro text
# --------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Alzheimer's Disease Predictor - G15</h1>", unsafe_allow_html=True)
st.write("Upload a CSV file with patient data to predict Alzheimer's Disease diagnosis.")

# --------------------------------------------------
# Load stacked model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return StackedXGBFNN.load('models')

model = load_model()

# --------------------------------------------------
# Session‑state helpers so UI widgets don't reset
# --------------------------------------------------
if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader("Choose your testing dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    st.session_state.test_data = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data:")
    st.dataframe(st.session_state.test_data.head())

    # --------------------------------------------------
    # Prediction button
    # --------------------------------------------------
    if st.button("Predict Alzheimer's Diagnosis"):
        try:
            preds = model.predict(st.session_state.test_data)
            probas = model.predict_proba(st.session_state.test_data)

            out = st.session_state.test_data.copy()
            out['Prediction'] = np.where(preds == 1, 'AD', 'Non-AD')
            out['Probability_AD'] = probas

            st.session_state.output_df = out  # store for later use
            st.success("Predictions generated successfully – scroll down for details.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------------------------------
# Show predictions if available
# --------------------------------------------------
if st.session_state.output_df is not None:
    st.markdown("## Prediction Results")
    st.dataframe(st.session_state.output_df)

    # Download button
    csv = st.session_state.output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download predictions as CSV",
        data=csv,
        file_name="alzheimers_predictions.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### SHAP Feature Contribution")

    # Record selector & checkbox live outside the prediction button to persist state
    record_idx = st.number_input(
        "Select record index to explain:",
        min_value=0,
        max_value=len(st.session_state.output_df) - 1,
        value=0,
        step=1,
        key="record_idx"
    )
    show_shap = st.checkbox("Show SHAP plot for selected record", key="show_shap")

    if show_shap:
        try:
            # preprocess the selected record only once
            selected_raw = st.session_state.test_data.iloc[[record_idx]]
            X_proc = model.preprocess_pipe.transform(selected_raw)

            # Use XGBoost part for SHAP (fast and tree‑based)
            explainer = shap.Explainer(model.xgb_model)
            shap_vals = explainer(X_proc)

            vals = shap_vals[0].values
            feat_names = model.preprocess_pipe.get_feature_names_out()
            sort_idx = np.argsort(np.abs(vals))[::-1]
            sorted_feats = feat_names[sort_idx]
            sorted_vals = vals[sort_idx]

            fig = go.Figure(go.Bar(
                x=sorted_vals,
                y=sorted_feats,
                orientation='h',
                marker_color=['green' if v > 0 else 'red' for v in sorted_vals]
            ))
            fig.update_layout(
                title=f"SHAP Contributions for Record {record_idx}",
                xaxis_title="SHAP Value (Impact on Log‑Odds)",
                yaxis_title="Feature",
                template="plotly_white",
                height=650
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
