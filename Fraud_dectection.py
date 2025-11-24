import streamlit as st
import pandas as pd
import joblib


try:
    model = joblib.load('fraud_detection_model.pkl')
except Exception as e:
    st.error(f"Could not load model: {e}")
    model = None

st.title("Fraud Detection App")

st.markdown("Please enter the transaction details below:")
# Use a simple horizontal rule for broad Streamlit compatibility
st.markdown("---")

transaction_type = st.selectbox(
    "Transaction Type",
    ["Payment", "Transfer", "Cash Out", "Debit", "Cash In"]
)

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=0.01)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, step=0.01)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, step=0.01, value=10000.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, step=0.01)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, step=0.01)

input_data = pd.DataFrame({
    'type': [transaction_type],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest]
})

if model is not None:
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        prediction = None

    if prediction is not None:
        st.subheader("Prediction Result:")
        st.write("Fraudulent Transaction" if int(prediction) == 1 else "Legitimate Transaction")
else:
    st.info("Model not available. Please provide a valid `fraud_detection_model.pkl` file in the app directory.")