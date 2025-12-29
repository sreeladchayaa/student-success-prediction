import pandas as pd
import numpy as np
import joblib
import os

PRED_HISTORY_CSV = "data/prediction_history.csv"

# Load trained model and encoders
rf = joblib.load("artifacts/rf_model.pkl")
le_skill = joblib.load("artifacts/le_skill.pkl")
le_success = joblib.load("artifacts/le_success.pkl")

feature_names = [
    "cgpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "skill_level",
    "projects_completed",
    "weekly_study_hours",
    "lms_login_frequency"
]

# For risk progression order mapping
risk_order = {"Low": 1, "Medium": 2, "High": 3}
risk_readable = {"Low": "At Risk", "Medium": "Moderate Risk", "High": "On Track"}

# Accept student_id and week_number as inputs
student_id = input("Enter student ID: ").strip()
week_number = input("Enter week number: ").strip()

# Example student input (can be replaced with user input if desired)
student_input = {
    "cgpa": 7.3,
    "attendance_percentage": 84,
    "assignment_completion_rate": 80,
    "skill_level": "Intermediate",
    "projects_completed": 3,
    "weekly_study_hours": 17,
    "lms_login_frequency": 7
}

# Prepare input DataFrame
input_df = pd.DataFrame([student_input])

# Encode categorical features
input_df["skill_level"] = le_skill.transform(input_df["skill_level"])

# Predict probabilities and class
pred_probs = rf.predict_proba(input_df)[0]
pred_class_idx = np.argmax(pred_probs)
pred_class_name = le_success.inverse_transform([pred_class_idx])[0]
risk_category = risk_readable.get(pred_class_name, pred_class_name)
success_probability = float(pred_probs[pred_class_idx])

# Identify top-3 most influential features (simple method as before)
importances = rf.feature_importances_
values = input_df.iloc[0].values
influence = np.abs(values * importances)
top3_idx = np.argsort(influence)[::-1][:3]
top3_features = [(feature_names[idx], values[idx], importances[idx]) for idx in top3_idx]

# Append prediction to prediction history
log_data = {
    "student_id": student_id,
    "week_number": week_number,
    "success_probability": f"{success_probability:.4f}",
    "risk_level": risk_category,
}
log_df = pd.DataFrame([log_data])

file_exists = os.path.isfile(PRED_HISTORY_CSV)
if not file_exists:
    log_df.to_csv(PRED_HISTORY_CSV, mode="w", index=False)
else:
    log_df.to_csv(PRED_HISTORY_CSV, mode="a", index=False, header=False)

# Load past prediction history for the same student (excluding this latest prediction)
if os.path.isfile(PRED_HISTORY_CSV):
    history = pd.read_csv(PRED_HISTORY_CSV)
    # Filter for this student_id and weeks < current
    history_student = history[
        (history["student_id"] == student_id) &
        (history["week_number"].astype(str) < str(week_number))
    ]
    # Sort by week_number descending to find latest past prediction
    if not history_student.empty:
        history_student_sorted = history_student.sort_values(
            by="week_number", ascending=False
        )
        previous_risk = history_student_sorted.iloc[0]["risk_level"]

        # Translate risk back to base risk class for order comparison
        # (if older log used readable "At Risk", "Moderate Risk", "On Track")
        def risk_to_order(risk):
            risk = str(risk)
            for k, v in risk_readable.items():
                if risk == v:
                    return risk_order[k]
            # fallback: try matching directly keys
            if risk in risk_order:
                return risk_order[risk]
            return None

        prev_risk_score = risk_to_order(previous_risk)
        curr_risk_score = risk_to_order(risk_category)

        # If risk increased (numerically moves from lower to higher: e.g. score 1->2 or 2->3)
        if prev_risk_score is not None and curr_risk_score is not None:
            if curr_risk_score > prev_risk_score:
                print("\n" + "!"*10 + " ALERT " + "!"*10)
                print(
                    f"Risk level increased for student '{student_id}': "
                    f"{previous_risk} (Week {history_student_sorted.iloc[0]['week_number']}) "
                    f"→ {risk_category} (Week {week_number})."
                )
                print("Recommend immediate support for this student.")
                print("!"*30 + "\n")

# Display results
print("="*45)
print("Student Success Prediction")
print("-" * 45)
print(f"Student ID                : {student_id}")
print(f"Week Number               : {week_number}")
print(f"Predicted Success Category: {pred_class_name}")
print(f"Risk Category             : {risk_category}")
print(f"Success Probabilities     :")
for idx, cls in enumerate(le_success.classes_):
    print(f"  {cls}: {pred_probs[idx]*100:.2f}%")
print("\nTop 3 Most Influential Features for This Student:")
for i, (fname, fval, fimp) in enumerate(top3_features, 1):
    print(f"  {i}. {fname}: {fval}  (Model Importance: {fimp:.4f})")
print("="*45)


