"""
app.py
------
Streamlit frontend for the AI Adoption Productivity Predictor.

Run with:
    streamlit run app.py

By default it calls the FastAPI service at http://127.0.0.1:8000.
Change API_URL below (or set the FASTAPI_URL environment variable) if your
API runs elsewhere (e.g. a deployed URL).
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Productivity Predictor", page_icon="🤖", layout="centered")

st.title("🤖 AI Adoption → Productivity Gain Predictor")
st.write(
    "Fill in a user's AI-adoption profile below and get a predicted "
    "**Productivity Gain (%)** from the trained model."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        industry = st.text_input("Industry", value="Technology")
        job_role = st.text_input("Job Role", value="Software Engineer")
        location = st.text_input("Location", value="Remote")
        primary_ai_tool = st.text_input("Primary AI Tool", value="ChatGPT")

    with col2:
        experience_years = st.number_input("Experience (years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
        daily_token_usage = st.number_input("Daily Token Usage", min_value=0.0, value=15000.0, step=100.0)
        tasks_automated = st.number_input("Tasks Automated Per Week", min_value=0.1, value=12.0, step=1.0)

    submitted = st.form_submit_button("Predict Productivity Gain")

if submitted:
    payload = {
        "Industry": industry,
        "Job_Role": job_role,
        "Location": location,
        "Primary_AI_Tool": primary_ai_tool,
        "Experience_Years": experience_years,
        "Daily_Token_Usage": daily_token_usage,
        "Tasks_Automated_Per_Week": tasks_automated,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()["predicted_productivity_gain_percent"]
            st.success(f"Predicted Productivity Gain: **{result}%**")
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the FastAPI service at `{API_URL}`. "
            "Make sure it's running (`uvicorn main:app --reload`)."
        )
    except requests.exceptions.Timeout:
        st.error("Request to the API timed out. Please try again.")

st.caption(f"Calling FastAPI at: {API_URL}")
