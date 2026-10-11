import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")
ARTIFACT = Path(__file__).with_name("student_performance_model.joblib")
if not ARTIFACT.exists():
    st.error("Model file is missing. Run: python train_model.py")
    st.stop()
pack = joblib.load(ARTIFACT)
model, features = pack["model"], pack["features"]

st.markdown("""
<style>
.block-container{max-width:1120px;padding-top:2rem}
.hero{padding:28px 30px;border-radius:22px;background:linear-gradient(120deg,#102a43,#1d4e89);color:white;margin-bottom:20px}
.hero h1{font-size:2.25rem;margin:0 0 8px 0}.hero p{color:#d8e8fa;margin:0;font-size:1.05rem}
div[data-testid="stMetric"]{background:var(--secondary-background-color);padding:18px;border-radius:15px;border:1px solid rgba(128,128,128,.18)}
</style>
<div class="hero"><h1>🎓 Student Performance Predictor</h1><p>Understand learning patterns, estimate academic outcomes, and discover practical ways to improve.</p></div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.subheader("Student learning profile")
    with st.form("predict_form"):
        c1,c2 = st.columns(2)
        with c1:
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
            previous_score = st.slider("Previous exam score (%)", 0, 100, 70)
        with c2:
            assignment_completion = st.slider("Assignments completed (%)", 0, 100, 80)
            sleep_hours = st.slider("Sleep per night (hours)", 0.0, 12.0, 7.0, 0.5)
            extracurricular_hours = st.slider("Extracurricular hours per day", 0.0, 8.0, 1.0, 0.5)
        submitted = st.form_submit_button("Predict performance", type="primary", use_container_width=True)

with right:
    st.subheader("How it works")
    st.write("A Random Forest regression model combines attendance, study habits, previous scores, assignment completion, sleep, and extracurricular time to estimate a score out of 100.")
    st.info("This educational demo was trained on synthetic data, not real student records. Its estimates are illustrative and should not be used for grading or high-stakes decisions.")

if submitted:
    row = pd.DataFrame([{"attendance":attendance,"study_hours":study_hours,"previous_score":previous_score,"assignment_completion":assignment_completion,"sleep_hours":sleep_hours,"extracurricular_hours":extracurricular_hours}])[features]
    score = float(model.predict(row)[0])
    score = max(0, min(100, score))
    st.divider()
    st.subheader("Prediction results")
    a,b,c = st.columns(3)
    a.metric("Estimated score", f"{score:.1f}/100")
    b.metric("Performance band", "Strong" if score >= 75 else "Developing" if score >= 50 else "Needs support")
    c.metric("Model validation MAE", f"{pack['mae']:.1f} points")
    st.progress(int(round(score)), text=f"Estimated score: {score:.1f}%")
    suggestions=[]
    if attendance < 85: suggestions.append("Aim for more consistent attendance, where possible.")
    if study_hours < 2: suggestions.append("Try a manageable daily study routine with short focused sessions.")
    if assignment_completion < 80: suggestions.append("Break assignments into smaller tasks and track deadlines.")
    if sleep_hours < 7: suggestions.append("Protect a regular, sufficient sleep schedule.")
    if previous_score < 60: suggestions.append("Review earlier mistakes and ask for help on topics that remain unclear.")
    if not suggestions: suggestions.append("Keep reviewing consistently and maintain a balanced routine.")
    st.markdown("#### Suggested next steps")
    for item in suggestions: st.write("- " + item)
    st.caption("A prediction is not a judgment of ability. Actual results depend on many factors the model does not capture.")

with st.expander("Model details"):
    st.write("Algorithm: Random Forest Regressor · Task: score estimation · Training examples: 6,000 synthetic records · Split: 80% train / 20% test")
    st.write(f"Holdout R² on synthetic data: {pack['r2']:.3f}. This metric only describes performance on the generated demo dataset.")
    st.download_button("Download sample dataset (CSV)", data=Path(__file__).with_name("synthetic_student_data.csv").read_bytes(), file_name="synthetic_student_data.csv", mime="text/csv")
