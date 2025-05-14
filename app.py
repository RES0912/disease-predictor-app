import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("catboost_disease_model.pkl")

model = load_model()

def predict_disease(input_dict):
    df = pd.DataFrame(input_dict, index=[0])
    prediction = model.predict(df)[0]
    confidence = max(model.predict_proba(df)[0]) * 100
    return prediction, confidence

def plot_feature_importance(model, feature_names):
    importances = model.get_feature_importance()
    plt.figure(figsize=(8, 4))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    st.pyplot(plt.gcf())


st.markdown("Enter values or upload a CSV to predict possible blood-related diseases.")

st.header("Manual Input")
inputs = {}
features = [
    "hematocrit", "hemoglobin", "mch", "mchc", "mcv", "wbc", 
    "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils"
]

defaults = [45, 15, 30, 35, 90, 7, 60, 30, 6, 2, 1]

for name, default in zip(features, defaults):
    inputs[name] = st.number_input(name, 0.0, 100.0, float(default), 0.1)

if st.button("Predict Disease"):
    pred, conf = predict_disease(inputs)
    st.success(f"Predicted Disease: {pred}")
    st.info(f"Confidence: {conf:.2f}%")

    st.markdown(" Feature Importance")
    plot_feature_importance(model, features)

    st.markdown(" Download Your Prediction")
    result_df = pd.DataFrame(inputs, index=[0])
    result_df["Prediction"] = pred
    result_df["Confidence"] = f"{conf:.2f}%"
    csv = result_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "prediction_result.csv")

st.divider()

st.header(" Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file with 11 features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    preds = model.predict(df)
    probs = model.predict_proba(df).max(axis=1) * 100

    df["Prediction"] = preds
    df["Confidence"] = [f"{p:.2f}%" for p in probs]

    st.markdown(" Results")
    st.dataframe(df)

    batch_csv = df.to_csv(index=False)
    st.download_button("Download Results CSV", batch_csv, "batch_predictions.csv")
