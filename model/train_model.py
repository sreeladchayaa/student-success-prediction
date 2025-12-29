import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv('data/student_data.csv')

# Encode categorical variables
le_skill = LabelEncoder()
le_success = LabelEncoder()
data['skill_level'] = le_skill.fit_transform(data['skill_level'])
data['success_level'] = le_success.fit_transform(data['success_level'])

# Print target variable distribution
print("Target Variable ('success_level') Distribution:")
success_counts = data['success_level'].value_counts()
for idx, count in success_counts.items():
    print(f"{le_success.inverse_transform([idx])[0]}: {count}")
print()

# Features and target
X = data.drop('success_level', axis=1)
y = data['success_level']

# Split into train (70%), validation (15%), and test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=42, stratify=y_temp
)
# Note: 0.17647 x 0.85 ≈ 0.15 for validation

# Logistic Regression with class_weight="balanced"
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

def print_metrics(y_true, y_pred, model_name, data_split):
    print(f"{model_name} - {data_split} Metrics:")
    print("  Accuracy :", accuracy_score(y_true, y_pred))
    print("  Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("  Recall   :", recall_score(y_true, y_pred, average='weighted'), "<-- KEY METRIC for At-Risk Students")
    print("  F1-score :", f1_score(y_true, y_pred, average='weighted'))
    print("  Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("  Classification Report:\n", classification_report(y_true, y_pred, target_names=le_success.classes_))
    print("-"*50)

# Evaluate on Validation Data
y_val_pred_lr = lr.predict(X_val)
y_val_pred_rf = rf.predict(X_val)

print("="*20, "VALIDATION SET", "="*20)
print_metrics(y_val, y_val_pred_lr, "Logistic Regression", "Validation")
print_metrics(y_val, y_val_pred_rf, "Random Forest Classifier", "Validation")

# Evaluate on Test Data
y_test_pred_lr = lr.predict(X_test)
y_test_pred_rf = rf.predict(X_test)

print("\n" + "="*22, "TEST SET", "="*22)
print_metrics(y_test, y_test_pred_lr, "Logistic Regression", "Test")
print_metrics(y_test, y_test_pred_rf, "Random Forest Classifier", "Test")

# ==============================
# RANDOM FOREST FEATURE IMPORTANCE ANALYSIS
# ==============================

importances = rf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
sorted_names = feature_names[indices]
sorted_importances = importances[indices]

# Create and display bar chart of feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), sorted_importances, align="center")
plt.xticks(range(len(importances)), sorted_names, rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances for Student Success Prediction")
plt.tight_layout()
plt.show()

# Print the top 5 most important features
print("\nTop 5 Most Important Features for Student Success (by Random Forest):")
for i in range(5):
    print(f"{i+1}. {sorted_names[i]} (Importance: {sorted_importances[i]:.4f})")

# EXPLANATION COMMENTS:
print("\nFeature Influence Explanations:")
for i in range(5):
    feature = sorted_names[i]
    print(f"- '{feature}': This feature is among the strongest predictors.")
    if feature == "cgpa":
        print("    -- Higher cumulative GPAs directly reflect overall academic performance, which strongly correlates with success.")
    elif feature == "attendance_percentage":
        print("    -- Students with high attendance are more likely to engage in learning and retain course material, boosting success.")
    elif feature == "assignment_completion_rate":
        print("    -- Consistently completing assignments demonstrates responsibility and understanding, which supports student achievement.")
    elif feature == "skill_level":
        print("    -- Advanced skill levels suggest better preparedness and coping strategies, increasing chance of success.")
    elif feature == "projects_completed":
        print("    -- Completing more projects can build practical experience and confidence, both contributing positively to outcomes.")
    elif feature == "weekly_study_hours":
        print("    -- More weekly study hours signal greater effort invested, which typically yields better results.")
    elif feature == "lms_login_frequency":
        print("    -- Frequent LMS logins indicate strong engagement with course resources, a factor in positive student outcomes.")

# ==============================
# SAVE ARTIFACTS (MODELS, ENCODERS, ETC)
# ==============================
artifacts_dir = 'artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

# Save the trained Random Forest model
rf_model_path = os.path.join(artifacts_dir, 'rf_model.pkl')
joblib.dump(rf, rf_model_path)

# Save encoders needed for prediction
le_skill_path = os.path.join(artifacts_dir, 'le_skill.pkl')
le_success_path = os.path.join(artifacts_dir, 'le_success.pkl')
joblib.dump(le_skill, le_skill_path)
joblib.dump(le_success, le_success_path)

print(f"\nSaved Random Forest model to {rf_model_path}")
print(f"Saved LabelEncoders to {le_skill_path} and {le_success_path}")



