import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Student Success Prediction System",
    page_icon="🎓",
    layout="wide",
)

# -------------------------------------------------
# GLOBAL STYLES
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #f5f7fb; }

h1, h2, h3, h4 {
    color: #0f172a !important;
    font-weight: 700;
}

p, label, span, div {
    color: #1f2937 !important;
}

.card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}

.card-low { border-left: 8px solid #dc2626; background-color: #fef2f2; }
.card-medium { border-left: 8px solid #f59e0b; background-color: #fffbeb; }
.card-high { border-left: 8px solid #16a34a; background-color: #f0fdf4; }

button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

input, textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Selectbox FIX */
div[data-baseweb="select"] > div { background-color: #ffffff !important; }
div[data-baseweb="select"] span { color: #111827 !important; }
ul[role="listbox"] li { color: #111827 !important; background-color: #ffffff !important; }

thead tr th {
    background-color: #e5e7eb !important;
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
MODEL_PATH = "artifacts/rf_model.pkl"
LE_SKILL_PATH = "artifacts/le_skill.pkl"
LE_SUCCESS_PATH = "artifacts/le_success.pkl"
HISTORY_PATH = "data/prediction_history.csv"

FEATURES = [
    "cgpa", "attendance_percentage", "assignment_completion_rate",
    "skill_level", "projects_completed",
    "weekly_study_hours", "lms_login_frequency"
]

SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"]
RISK_LABELS = {"Low": "At Risk", "Medium": "Moderate Risk", "High": "On Track"}
RISK_COLORS = {"Low": "#dc2626", "Medium": "#f59e0b", "High": "#16a34a"}

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(LE_SKILL_PATH),
        joblib.load(LE_SUCCESS_PATH),
    )

def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["student_id", "week", "probability", "risk", "time"])

def save_history(row):
    os.makedirs("data", exist_ok=True)
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("## 🎓 Student Success Prediction System")
st.markdown(
    "<p style='color:#475569;'>Early Risk Detection • Explainable AI • Institutional Monitoring</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["🔮 Single Student", "📂 Bulk CSV", "📈 Risk Timeline"])

# -------------------------------------------------
# TAB 1 — SINGLE STUDENT
# -------------------------------------------------
with tabs[0]:
    st.subheader("Single Student Prediction")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            student_id = st.text_input("Student ID")
            week = st.number_input("Week Number", min_value=1)

        with c2:
            cgpa = st.number_input("CGPA", 0.0, 10.0)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0)
            assignment = st.number_input("Assignment Completion (%)", 0.0, 100.0)

        with c3:
            skill = st.selectbox("Skill Level", SKILL_LEVELS)
            projects = st.number_input("Projects Completed", 0)
            study_hours = st.number_input("Weekly Study Hours", 0.0)
            lms = st.number_input("LMS Logins / Week", 0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        rf, le_skill, le_success = load_artifacts()

        X = pd.DataFrame([{
            "cgpa": cgpa,
            "attendance_percentage": attendance,
            "assignment_completion_rate": assignment,
            "skill_level": le_skill.transform([skill])[0],
            "projects_completed": projects,
            "weekly_study_hours": study_hours,
            "lms_login_frequency": lms,
        }])

        probs = rf.predict_proba(X)[0]
        idx = np.argmax(probs)
        risk = le_success.inverse_transform([idx])[0]
        prob = probs[idx]

        risk_class = (
            "card-low" if risk == "Low"
            else "card-medium" if risk == "Medium"
            else "card-high"
        )

        st.markdown(f"""
        <div class="card {risk_class}">
            <h4>Prediction Result</h4>
            <p><b>Student ID:</b> {student_id}</p>
            <p><b>Risk Level:</b>
               <span style="color:{RISK_COLORS[risk]};font-weight:600;">
               {RISK_LABELS[risk]}</span></p>
            <p><b>Success Probability:</b> {prob*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        save_history({
            "student_id": student_id,
            "week": week,
            "probability": prob,
            "risk": risk,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# -------------------------------------------------
# TAB 2 — BULK CSV (DOWNLOAD ADDED)
# -------------------------------------------------
with tabs[1]:
    st.subheader("Bulk CSV Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        rf, le_skill, le_success = load_artifacts()

        df["skill_level"] = le_skill.transform(df["skill_level"])
        X = df[FEATURES]

        probs = rf.predict_proba(X)
        preds = le_success.inverse_transform(np.argmax(probs, axis=1))

        df["Risk"] = [RISK_LABELS[r] for r in preds]
        df["Probability (%)"] = (probs.max(axis=1) * 100).round(2)

        st.dataframe(df, use_container_width=True)

        # 🔽 DOWNLOAD BUTTON (NEW)
        st.download_button(
            label="⬇ Download Predicted Results",
            data=df.to_csv(index=False),
            file_name="student_success_predictions.csv",
            mime="text/csv"
        )

# -------------------------------------------------
# TAB 3 — TIMELINE
# -------------------------------------------------
with tabs[2]:
    st.subheader("Student Risk Timeline")

    sid = st.text_input("Enter Student ID")
    if st.button("Show Timeline"):
        hist = load_history()
        hist = hist[hist["student_id"] == sid]

        if hist.empty:
            st.warning("No history found.")
        else:
            hist = hist.sort_values("week")
            fig, ax = plt.subplots()
            ax.plot(hist["week"], hist["probability"], marker="o")
            ax.set_xlabel("Week")
            ax.set_ylabel("Success Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b;'>Student Success Prediction System • SIH-Ready Project</p>",
    unsafe_allow_html=True,
)
