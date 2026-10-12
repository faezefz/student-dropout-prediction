import pandas as pd
import shap

from src.features import CATEGORICAL_COLS


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def _original_feature(name):
    """Map a transformed feature name back to its original column."""
    prefix, rest = name.split("__", 1)
    if prefix == "cat":
        for col in CATEGORICAL_COLS:
            if rest.startswith(col + "_"):
                return col
    return rest


class ModelExplainer:
    """Explain predictions of the Logistic Regression pipeline with SHAP."""

    def __init__(self, model, X_background):
        self.preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        background = _to_dense(self.preprocessor.transform(X_background))
        self.explainer = shap.LinearExplainer(classifier, background)
        self.base_value = self.explainer.expected_value

        feature_names = self.preprocessor.get_feature_names_out()
        self.original_names = [_original_feature(n) for n in feature_names]

    def explain(self, X_one):
        """Return SHAP contributions (log-odds) per original feature for one student."""
        X_t = _to_dense(self.preprocessor.transform(X_one))
        values = self.explainer(X_t).values[0]
        contributions = pd.Series(values, index=self.original_names)
        return contributions.groupby(level=0).sum().sort_values()