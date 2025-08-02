import streamlit as st
import pickle
import joblib
import numpy as np
from scipy.sparse import hstack
import pandas as pd
df=pd.read_csv("fake_jobs.csv")
# Set page configuration
st.set_page_config(page_title="Fake Job Posting Detector", layout="wide", page_icon="🕵️")

# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, tfidf, scaler, label_encoders

model, tfidf, scaler, label_encoders = load_resources()

# ---------- Page Background Style ---------- #
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1f4037, #99f2c8);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Landing Page ---------- #
st.markdown("""
    <div style='text-align:center; background: linear-gradient(to right, #2b5876, #4e4376); padding: 40px 20px; border-radius: 12px;'>
        <h1 style='color:#F1F1F1;'>🕵️ Fake Job Posting Classifier</h1>
        <h3 style='color:#D1D1D1;'>AI-Powered Detection for Safer Job Market</h3>
        <p style='color:#E1E1E1; max-width:900px; margin:auto; font-size:16px;'>
            This application uses advanced Natural Language Processing (NLP) techniques such as TF-IDF vectorization along with a Balanced Random Forest Classifier to detect fake job postings. 
            Structured features like company details, job type, and education requirements are scaled and used alongside job description analysis to identify fraud indicators effectively.
        </p>
        <img src='https://img.freepik.com/free-vector/cyber-fraud-illustration_114360-7884.jpg' width='15%' style='border-radius:12px; margin-top:20px;'>
    </div>
    <hr>
""", unsafe_allow_html=True)

# ---------- Input Form ---------- #
st.subheader("📝 Enter Job Posting Details")
col1, col2 = st.columns(2)

with col1:
    title = st.selectbox("Job Title", df["title"].unique())
    has_company_logo = st.radio("Has Company Logo?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True,width=552)
    has_questions = st.radio("Has Screening Questions?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True,width=552)
    employment_type = st.selectbox("Employment Type", label_encoders['employment_type'].classes_)
    required_experience = st.selectbox("Required Experience", label_encoders['required_experience'].classes_)

with col2:
    required_education = st.selectbox("Required Education", label_encoders['required_education'].classes_)
    industry = st.selectbox("Industry", label_encoders['industry'].classes_)
    company_profile = st.text_area("Company Profile", height=100)
    description = st.text_area("Job Description", height=100)
    requirements = st.text_area("Requirements", height=100)

# ---------- Prediction Section ---------- #
if st.button("🔍 Predict Fraud Probability"):
    with st.spinner("Analyzing the job posting🧐"):
        # Combine text inputs
        text = title + " " + company_profile + " " + description + " " + requirements
        X_text = tfidf.transform([text])

        # Encode categorical inputs
        cat_features = [
            has_company_logo,
            has_questions,
            label_encoders['employment_type'].transform([employment_type])[0],
            label_encoders['required_experience'].transform([required_experience])[0],
            label_encoders['required_education'].transform([required_education])[0],
            label_encoders['industry'].transform([industry])[0]
        ]
        cat_features_scaled = scaler.transform([cat_features])

        # Combine features
        final_input = hstack([X_text, cat_features_scaled])

        # Predict
       # In your prediction section, modify this line:
        prob = min(model.predict_proba(final_input)[0][1] * 100, 100.0)  # Cap at 100%
        label = model.predict(final_input)[0]

        # Output box with gradient-friendly colors
        result_color = "#E74C3C" if label == 1 else "#27AE60"
        result_text = "❗️ Fraudulent" if label == 1 else "✅ Legitimate"

        st.markdown(f"""
        <div style='background: linear-gradient(to right, #434343, #000000); padding:20px; border-radius:12px;'>
            <h4 style='color:#FFFFFF;'>Prediction Result</h4>
            <p style='color:#FFFFFF;'><strong>Fraud Probability:</strong> <span style='color:{result_color};'>{prob:.2%}</span></p>
            <p style='color:#FFFFFF;'><strong>Prediction:</strong> <span style='color:{result_color};'>{result_text}</span></p>
        </div>
        """, unsafe_allow_html=True)
import streamlit as st

# Set custom page config (if not already there)
st.set_page_config(page_title="Fake Job Posting Detector", layout="wide")

# Gradient background and Predict button hover effect
st.markdown(
    """
    <style>
    /* Gradient background for the whole page */
    .stApp {
        background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Optional: Enhance card-like input container appearance */
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Predict button hover effect only */
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        transition: 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #5a189a !important;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(90, 24, 154, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)
