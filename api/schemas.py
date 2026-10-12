from pydantic import BaseModel, Field


class StudentData(BaseModel):
    # Categorical (codes)
    marital_status: int = Field(ge=1, le=6)
    application_mode: int
    course: int
    nationality: int
    previous_qualification: int
    mother_s_qualification: int
    father_s_qualification: int
    mother_s_occupation: int
    father_s_occupation: int

    # Binary (0 or 1)
    gender: int = Field(ge=0, le=1)
    international: int = Field(ge=0, le=1)
    displaced: int = Field(ge=0, le=1) 
    educational_special_needs: int = Field(ge=0, le=1) 
    daytime_evening_attendance: int = Field(ge=0, le=1) 
    debtor: int = Field(ge=0, le=1)
    scholarship_holder: int = Field(ge=0, le=1)


    # Numerical
    age_at_enrollment: int = Field(ge=15, le=100)
    application_order: int = Field(ge=0, le=9)
    previous_qualification_grade: float = Field(ge=0, le=200)
    admission_grade: float = Field(ge=0, le=200)
    curricular_units_1st_sem_credited: int = Field(ge=0)
    curricular_units_1st_sem_enrolled: int = Field(ge=0)
    curricular_units_1st_sem_evaluations: int = Field(ge=0)
    curricular_units_1st_sem_approved: int = Field(ge=0)
    curricular_units_1st_sem_grade: float = Field(ge=0, le=20)
    curricular_units_1st_sem_without_evaluations: int = Field(ge=0)
    unemployment_rate: float
    inflation_rate: float
    gdp: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }