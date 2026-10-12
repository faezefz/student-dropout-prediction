import sys
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data import load_data  # noqa: E402
from src.features import get_X_y  # noqa: E402
from src.model import RANDOM_STATE, THRESHOLD, build_model  # noqa: E402
from src.explain import ModelExplainer  # noqa: E402

BASE_STUDENT = {
    "marital_status": 1.0,
    "application_mode": 1.0,
    "application_order": 2.0,
    "course": 9238.0,
    "daytime_evening_attendance": 1.0,
    "previous_qualification": 1.0,
    "previous_qualification_grade": 140.0,
    "nationality": 1.0,
    "mother_s_qualification": 19.0,
    "father_s_qualification": 37.0,
    "mother_s_occupation": 9.0,
    "father_s_occupation": 9.0,
    "admission_grade": 140.7,
    "displaced": 1.0,
    "educational_special_needs": 0.0,
    "debtor": 0.0,
    "gender": 0.0,
    "scholarship_holder": 1.0,
    "age_at_enrollment": 18.0,
    "international": 0.0,
    "curricular_units_1st_sem_credited": 0.0,
    "curricular_units_1st_sem_enrolled": 6.0,
    "curricular_units_1st_sem_evaluations": 6.0,
    "curricular_units_1st_sem_approved": 6.0,
    "curricular_units_1st_sem_grade": 13.166666666666666,
    "curricular_units_1st_sem_without_evaluations": 0.0,
    "unemployment_rate": 10.8,
    "inflation_rate": 1.4,
    "gdp": 1.74                       
}


@st.cache_resource
def get_model_and_explainer():
    df_model, _ = load_data(ROOT / "data" / "data.csv")
    X, y = get_X_y(df_model)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model()
    model.fit(X_train, y_train)
    explainer = ModelExplainer(model, X_train)
    return model, explainer


model, explainer = get_model_and_explainer()

st.title("🎓 Student Dropout Risk Predictor")
st.write(
    "Predicts the risk that a student will drop out, "
    "based on enrollment data and first-semester results."
)

age = st.slider("Age at enrollment", 17, 70, int(BASE_STUDENT["age_at_enrollment"]))
gender = st.radio("Gender", ["Female", "Male"], index=int(BASE_STUDENT["gender"]))
debtor = st.checkbox("Debtor (owes money to the university)",
                     value=BASE_STUDENT["debtor"] == 1)
scholarship = st.checkbox("Scholarship holder",
                          value=BASE_STUDENT["scholarship_holder"] == 1)
courses_enrolled = st.slider("Courses enrolled (1st semester)", 0, 20,
                              int(BASE_STUDENT["curricular_units_1st_sem_enrolled"]))
courses_approved = st.slider("Courses approved (1st semester)", 0, 20,
                              int(BASE_STUDENT["curricular_units_1st_sem_approved"]))
average_grade = st.slider("Average grade (1st semester, 0–20)", 0.0, 20.0,
                          float(round(BASE_STUDENT["curricular_units_1st_sem_grade"], 1)),
                          step=0.1)
admission_grade = st.slider("Admission grade (0–200)", 95.0, 200.0,
                            float(BASE_STUDENT["admission_grade"]),
                            step=0.1)

student = BASE_STUDENT.copy()
student.update({
    "age_at_enrollment": age,
    "gender": 1 if gender == "Male" else 0,
    "debtor": int(debtor),
    "scholarship_holder": int(scholarship),
    "curricular_units_1st_sem_enrolled": courses_enrolled,
    "curricular_units_1st_sem_approved": courses_approved,
    "curricular_units_1st_sem_grade": average_grade,
    "admission_grade": admission_grade
})

student_df = pd.DataFrame([student])     
proba = model.predict_proba(student_df)[0, 1]

st.metric("Dropout probability", f"{proba:.1%}")
if proba >= THRESHOLD:
    st.error("⚠️ At risk of dropping out")
else:
    st.success("✅ Not at risk")

st.subheader("Why this prediction?")

contributions = explainer.explain(student_df)
top = contributions.reindex(
    contributions.abs().sort_values(ascending=False).index
).head(8)[::-1]

labels = (
    top.index
    .str.replace("curricular_units_1st_sem_", "1st sem: ")
    .str.replace("_", " ")
)
colors = ["#d62728" if v > 0 else "#2ca02c" for v in top.values]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(labels, top.values, color=colors)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("<- lowers risk           raises risk ->")
st.pyplot(fig)

st.caption(
    "Red bars push the prediction towards dropout, while green bars push it towards graduation."
    "Values are SHAP contributions to the log-odds, compared to an average student."
)