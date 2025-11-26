# Fraud_detection.py
# Streamlit app for fraud detection (safe imports and graceful handling when joblib is missing)

import os
import sys
import warnings

import streamlit as st
import pandas as pd
import numpy as np

# Try to import joblib. If not available, try sklearn.externals.joblib (old sklearn) and otherwise mark as unavailable.
try:
    import joblib
except Exception:
    try:
        from sklearn.externals import joblib  # older sklearn versions
    except Exception:
        joblib = None

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("Fraud Detection")

MODEL_CANDIDATES = [
    "model.joblib",
    "model.pkl",
    "model.sav",
    "fraud_model.joblib",
    "fraud_model.pkl",
]


def find_model_path():
    """Search common model filenames in the app root and return the first that exists."""
    for name in MODEL_CANDIDATES:
        if os.path.exists(name):
            return name
    return None


def load_model(path=None):
    """Attempt to load a model using joblib (if available). Return None on failure.

    This function will not raise ImportError if joblib is missing; instead it shows
    a helpful message in the Streamlit app so the app doesn't crash with ModuleNotFoundError.
    """
    if joblib is None:
        st.warning(
            "Python package 'joblib' is not installed in the environment. "
            "Add 'joblib' to requirements.txt (e.g. joblib==1.2.0) and redeploy, or install it locally."
        )
        return None

    if path is None:
        path = find_model_path()

    if path is None:
        st.info("No model file found in the repository root. Expected one of: " + ", ".join(MODEL_CANDIDATES))
        return None

    try:
        model = joblib.load(path)
        st.success(f"Loaded model from {path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None


# Load the model (or None if missing / failed to load)
model = load_model()

# Example UI: if model is not available, we keep the app running but show instructions.
if model is None:
    st.subheader("Model not available")
    st.write(
        "The machine learning model is not available in this deployment. To fix this:\n"
        "1. Add the trained model file (for example `model.joblib`) to the repository root.\n"
        "2. Ensure your `requirements.txt` contains `joblib` (for example `joblib==1.2.0`) and any other dependencies like scikit-learn.\n"
        "3. Re-deploy the app on Streamlit Cloud."
    )
    # Optionally provide an upload widget so the user can upload a model at runtime
    uploaded_file = st.file_uploader("Upload a model file (joblib .joblib or .pkl)", type=["joblib", "pkl", "sav"]) 
    if uploaded_file is not None:
        try:
            # Try to load uploaded model using joblib if available, otherwise try pickle
            if joblib is not None:
                model = joblib.load(uploaded_file)
                st.success("Model loaded from uploaded file")
            else:
                import pickle

                model = pickle.load(uploaded_file)
                st.success("Model loaded from uploaded file using pickle")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

else:
    st.subheader("Model ready")
    st.write("The model is loaded and ready to use. Add your input controls below to run predictions.")

    # Example placeholder: you should adapt the inputs to match your model's expected features
    st.write("---")
    st.write("Example input form (customize to your model)")
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.number_input("Feature 1", value=0.0)
        feat2 = st.number_input("Feature 2", value=0.0)
    with col2:
        feat3 = st.number_input("Feature 3", value=0.0)
        feat4 = st.number_input("Feature 4", value=0.0)

    if st.button("Predict"):
        try:
            X = np.array([[feat1, feat2, feat3, feat4]])
            pred = model.predict(X)
            st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    # Nothing special to run; Streamlit runs this module when launching the app.
    pass
