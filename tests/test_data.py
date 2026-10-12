import pandas as pd

from src.data import clean_column_names, load_data

DATA_PATH = "data/data.csv"


def test_clean_column_names():
    df = pd.DataFrame(columns=["Marital status", "Daytime/evening attendance\t", "Nacionality"])
    result = clean_column_names(df)
    assert list(result.columns) == ["marital_status", "daytime_evening_attendance", "nationality"]


def test_load_data_removes_enrolled():
    df_model, df_enrolled = load_data(DATA_PATH)
    assert "Enrolled" not in df_model["target"].values
    assert (df_enrolled["target"] == "Enrolled").all()


def test_dropout_column_is_binary():
    df_model, _ = load_data(DATA_PATH)
    assert set(df_model["dropout"].unique()) == {0, 1}