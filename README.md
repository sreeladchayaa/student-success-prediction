# 🎓 Student Success Prediction System

An end-to-end machine learning system that predicts student success risk early using academic performance, engagement, and behavioral data, enabling proactive academic intervention through continuous monitoring and explainable AI.

---

## 📌 Project Overview

Educational institutions often identify struggling students only after visible failure such as low grades or poor attendance. This reactive approach limits timely support and increases dropout risk.

The **Student Success Prediction System** addresses this gap by predicting student success probability at an early stage. The system continuously tracks student performance over time, detects rising risk levels, and provides interpretable insights to support data-driven academic intervention.

This project is designed as a **real-world, institution-ready ML system**, not just a one-time prediction model.

---

## 🎯 Problem Statement

Educational institutions lack an intelligent, data-driven mechanism to identify students at risk of poor academic performance early using academic and engagement indicators.

As a result, interventions often happen too late. There is a need for a system that can **predict risk early, track changes over time, and explain the reasons behind predictions** to enable proactive support.

---

## 🚀 Key Features

- 🔮 **Single Student Prediction**
  - Predicts success probability and risk level for individual students
  - Clear visual risk indicators (At Risk / Moderate Risk / On Track)

- 📂 **Bulk CSV Prediction**
  - Upload class-level data
  - Predict risk for multiple students at once
  - Download prediction results as CSV

- ⏱ **Early Warning System**
  - Logs predictions weekly
  - Detects increasing risk trends
  - Enables timely academic intervention

- 📈 **Student Risk Timeline**
  - Visualizes success probability changes over time
  - Tracks improvement or deterioration in performance

- 🧠 **Explainability-Ready Design**
  - Designed to support Explainable AI (feature influence and SHAP-based interpretation)

- 🎛 **Interactive Dashboard**
  - Built using Streamlit with a clean, professional UI
  - Suitable for students, faculty, and institutional use

---

## 📊 Dataset & Features

The system uses structured student data including:

### Academic Indicators
- CGPA
- Attendance percentage
- Assignment completion rate

### Engagement Indicators
- Weekly study hours
- LMS login frequency

### Skill Indicators
- Skill level (Beginner / Intermediate / Advanced)
- Number of completed projects

---

## 🧠 Machine Learning Approach

- **Problem Type:** Supervised Classification
- **Model Used:** Random Forest Classifier
- **Why Random Forest?**
  - Handles non-linear relationships well
  - Robust to noise
  - Provides feature importance for interpretability

### Output
- Success probability score
- Risk category:
  - At Risk
  - Moderate Risk
  - On Track

---

## 🏗 System Architecture

1. Data preprocessing and encoding
2. Model training and evaluation
3. Model artifact storage
4. Interactive Streamlit UI
5. Prediction logging and
