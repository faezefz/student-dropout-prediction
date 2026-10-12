from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.model import RANDOM_STATE, THRESHOLD, build_model


from src.data import load_data
from src.features import get_X_y

DATA_PATH = "data/data.csv"
MODEL_PATH = Path("models/model.joblib")


def evaluate(model, X_test, y_test, threshold):
    """Evaluate the model on the test set using a custom threshold."""
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    # 1. Load data
    df_model, _ = load_data(DATA_PATH)
    X, y = get_X_y(df_model)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

    # 3. Train
    model = build_model()
    model.fit(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate(model, X_test, y_test, THRESHOLD)
    print(metrics)

    # 5. Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student-dropout")
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metrics(metrics)

    # 6. Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": THRESHOLD}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()