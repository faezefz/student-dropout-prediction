
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

THRESHOLD = 0.3
RANDOM_STATE = 42

def build_model():
    """Create the full pipeline: preprocessing + classifier."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])