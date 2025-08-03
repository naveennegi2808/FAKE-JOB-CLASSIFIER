import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
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

# ---------- Sidebar for Threshold Control ---------- #
st.sidebar.header("⚙️ Detection Sensitivity")
threshold = st.sidebar.slider(
    "Fraud Detection Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.50,
    step=0.05,
    help="Higher values = stricter detection (fewer false alarms, might miss some fraud)"
)

st.sidebar.markdown("""
**How it works:** The threshold determines when a job posting gets flagged as fraudulent. 
- **Low (10-40%):** Catches more fraud but may flag legitimate jobs
- **Medium (50-60%):** Balanced approach - recommended setting
- **High (70-90%):** Very strict - only flags obvious fraud
""")

# ---------- Input Form ---------- #
st.subheader("📝 Enter Job Posting Details")
col1, col2 = st.columns(2)

with col1:
    title = st.selectbox("Job Title", df['title'].unique(), help="Select a job title from the list")
    has_company_logo = st.radio("Has Company Logo?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True,width=600)
    has_questions = st.radio("Has Screening Questions?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True,width=600)
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
    with st.spinner("Analyzing the job post..."):
        try:
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

            # Get probability and apply threshold
            prob = model.predict_proba(final_input)[0]
            fraud_prob = prob[1]  # Probability of being fraudulent
            legit_prob = prob[0]  # Probability of being legitimate
            label = 1 if fraud_prob > threshold else 0

            # Suspicious word analysis
            suspicious_words = [
            # Original list
            "urgent", "hiring", "bonus", "fee", "whatsapp", "immediately", 
            "guaranteed", "processing fee", "no experience", "easy money",
            "cash app", "venmo", "wire transfer", "send money", "pay now",
            "limited time", "act fast", "exclusive offer", "secret code",
    
            # High-confidence suspicious additions (very likely scam indicators)
            "upfront payment", "startup fee", "training fee", "deposit required",
            "western union", "money gram", "bitcoin", "social security number",
            "make money fast", "earn thousands weekly", "work from home guaranteed",
            "no skills required", "anyone can do", "envelope stuffing",
            "mystery shopper", "data entry from home", "text only interview",
            "personal email", "gmail interview", "yahoo hiring", "hotmail contact",
            "po box address", "international company urgent", "government certified opportunity",
            "recruit others", "build your team", "copy of id required",
            "send photo first", "background check fee", "equipment purchase required"
]
            found_suspicious = [word for word in suspicious_words if word.lower() in text.lower()]

            # Output box with gradient-friendly colors
            result_color = "#E74C3C" if label == 1 else "#27AE60"
            result_text = "❗️ Fraudulent" if label == 1 else "✅ Legitimate"

            st.markdown(f"""
            <div style='background: linear-gradient(to right, #434343, #000000); padding:20px; border-radius:12px; margin-top:20px;'>
                <h4 style='color:#FFFFFF;'>📊 Analysis Results</h4>
                <p style='color:#FFFFFF;'><strong>Final Prediction:</strong> <span style='color:{result_color}; font-size:18px;'>{result_text}</span></p>
                <div style='display:flex; justify-content:space-between; margin:15px 0;'>
                    <div style='color:#27AE60;'>✅ Legitimate: {legit_prob:.1%}</div>
                    <div style='color:#E74C3C;'>❗️ Fraudulent: {fraud_prob:.1%}</div>
                </div>
                <p style='color:#FFFFFF;'><strong>Threshold Used:</strong> {threshold:.0%} - {'Strict Detection' if threshold > 0.7 else 'Balanced Detection' if threshold > 0.4 else 'Sensitive Detection'}</p>
            </div>
            """, unsafe_allow_html=True)

            # Suspicious words section
            if found_suspicious:
                st.markdown("""
                <div style='background: linear-gradient(to right, #8B0000, #DC143C); padding:15px; border-radius:8px; margin:10px 0;'>
                    <h5 style='color:#FFFFFF; margin:0;'>🚨 Suspicious Language Detected</h5>
                </div>
                """, unsafe_allow_html=True)
                
                suspicious_text = ", ".join([f"**{word}**" for word in found_suspicious])
                st.warning(f"⚠️ Found suspicious words/phrases: {suspicious_text}")
            else:
                st.info("✅ No obvious suspicious language patterns detected")

            # Context message
            if label == 1:
                context_msg = "⚠️ **High Risk:** This job posting shows signs of fraud. Verify company details independently and never send money or personal documents upfront."
            else:
                if fraud_prob > 0.3:
                    context_msg = "⚠️ **Moderate Risk:** While classified as legitimate, some concerning elements detected. Exercise caution and verify company details."
                else:
                    context_msg = "✅ **Low Risk:** This job posting appears legitimate. Always verify company details before sharing personal information."

            if label == 1:
                st.error(context_msg)
            elif fraud_prob > 0.3:
                st.warning(context_msg)
            else:
                st.success(context_msg)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check your inputs and try again.")

# ---------- Advanced Styling ---------- #
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
    
    /* Style the sidebar */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)