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