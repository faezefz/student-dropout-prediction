from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

# Works both locally (models/model.joblib) and on Hugging Face (model.joblib)
LOCAL_PATH = Path("models/model.joblib")
MODEL_PATH = LOCAL_PATH if LOCAL_PATH.exists() else Path("model.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]

# A real student from the test set, used as default values
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


def predict(age, gender, debtor, scholarship_holder,
            units_enrolled, units_approved, grade, admission_grade):
    student = BASE_STUDENT.copy()
    student.update({
        "age_at_enrollment": age,
        "gender": 1 if gender == "Male" else 0,
        "debtor": int(debtor),
        "scholarship_holder": int(scholarship_holder),
        "curricular_units_1st_sem_enrolled": units_enrolled,
        "curricular_units_1st_sem_approved": units_approved,
        "curricular_units_1st_sem_grade": grade,
        "admission_grade": admission_grade,
    })

    df = pd.DataFrame([student])
    proba = model.predict_proba(df)[0, 1]

    label = "⚠️ At risk of dropping out" if proba >= threshold else "✅ Not at risk"
    return f"{proba:.1%}", label


demo = gr.Interface(
    flagging_mode="never",
    fn=predict,
    inputs=[
        gr.Slider(17, 70, value=BASE_STUDENT["age_at_enrollment"], step=1, label="Age at enrollment"),
        gr.Radio(["Female", "Male"], value="Female" if BASE_STUDENT["gender"] == 0 else "Male", label="Gender"),
        gr.Checkbox(value=BASE_STUDENT["debtor"]==1, label="Debtor (owes money to the university)"),
        gr.Checkbox(value=BASE_STUDENT["scholarship_holder"]==1, label="Scholarship holder"),
        gr.Slider(0, 20, value=round(BASE_STUDENT["curricular_units_1st_sem_enrolled"], 1), step=1, label="Cources enrolled (1st semester)"),
        gr.Slider(0, 20, value=round(BASE_STUDENT["curricular_units_1st_sem_approved"], 1), step=1, label="Cources approved (1st semester)"),
        gr.Slider(0, 20, value=round(BASE_STUDENT["curricular_units_1st_sem_grade"], 1), step=0.1, label="Cources grade (1st semester)"),   
        gr.Slider(95, 200, value=round(BASE_STUDENT["admission_grade"], 1), step=0.1, label="Admission grade"),
    ],
    outputs=[
        gr.Textbox(label="Dropout probability"),
        gr.Textbox(label="Prediction"),
    ],
    title="🎓 Student Dropout Risk Predictor",
    description=(
        "Predicts the risk that a student will drop out, "
        "based on enrollment data and first-semester results."
    ),
)

if __name__ == "__main__":
    demo.launch()