from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "dropout"

CATEGORICAL_COLS = [
    "marital_status", 
    "nationality", 
    "application_mode", 
    "course", 
    "previous_qualification", 
    "mother_s_qualification", 
    "father_s_qualification", 
    "mother_s_occupation",
    "father_s_occupation"
]

BINARY_COLS = [
    "gender", 
    "international", 
    "displaced", 
    "educational_special_needs", 
    "daytime_evening_attendance", 
    "debtor", 
    "scholarship_holder"
]

NUMERICAL_COLS = [
    "age_at_enrollment", 
    "application_order", 
    "previous_qualification_grade", 
    "admission_grade", 
    "curricular_units_1st_sem_credited", 
    "curricular_units_1st_sem_enrolled", 
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved", 
    "curricular_units_1st_sem_grade", 
    "curricular_units_1st_sem_without_evaluations", 
    "unemployment_rate", 
    "inflation_rate", 
    "gdp"
]

FEATURE_COLS = CATEGORICAL_COLS + BINARY_COLS + NUMERICAL_COLS


def build_preprocessor():
    """Create the preprocessing step for the model pipeline."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )


def get_X_y(df):
    """Select feature columns and target column from the dataframe."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    from src.data import load_data

    df_model, _ = load_data("data/data.csv")
    X, y = get_X_y(df_model)
    print(X.shape, y.shape)