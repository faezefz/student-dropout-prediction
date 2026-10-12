import numpy as np
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.explain import ModelExplainer
from src.features import FEATURE_COLS, get_X_y
from src.model import RANDOM_STATE, build_model


def test_shap_values_add_up_to_prediction():
    df_model, _ = load_data("data/data.csv")
    X, y = get_X_y(df_model)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model()
    model.fit(X_train, y_train)

    explainer = ModelExplainer(model, X_train)
    student = X_test.iloc[[0]]
    contributions = explainer.explain(student)

    # One contribution per original feature
    assert len(contributions) == len(FEATURE_COLS)

    # SHAP values add up to the model's prediction
    log_odds = explainer.base_value + contributions.sum()
    prob_from_shap = 1 / (1 + np.exp(-log_odds))
    prob_from_model = model.predict_proba(student)[0, 1]
    assert np.isclose(prob_from_shap, prob_from_model)