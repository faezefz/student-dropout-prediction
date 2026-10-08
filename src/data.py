import pandas as pd


def clean_column_names(df):
    """Convert column names to snake_case and fix typos."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df = df.rename(columns={"nacionality": "nationality"})
    return df


def load_data(path):
    """
    Load the dataset and split it into two parts:
    - df_model: Dropout and Graduate students, with a binary 'dropout' column
    - df_enrolled: Enrolled students (outcome unknown)
    """
    df = pd.read_csv(path, sep=";")
    df = clean_column_names(df)

    df_model = df[df["target"] != "Enrolled"].copy()
    df_model["dropout"] = (df_model["target"] == "Dropout").astype(int)

    df_enrolled = df[df["target"] == "Enrolled"].copy()

    return df_model, df_enrolled


if __name__ == "__main__":
    df_model, df_enrolled = load_data("data/data.csv")
    print(df_model.shape, df_enrolled.shape)