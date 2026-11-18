"""Model zoo: the 5+ classifiers we compare. OWNER: Kushal.

Every model must support predict_proba. Scalers live inside a sklearn Pipeline so they
are fitted on training folds only. config.SEED is used for every random_state.

The task for every model: given one (GPS point, candidate arc) row of features,
output P(this candidate is the arc the car was really on).
"""
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import SEED


def logistic_regression(C=1.0, class_weight=None):
    """Linear model: P(correct arc) = sigmoid(w . scaled_features + b).

    First model: fast, usually well-behaved probabilities (good for the HMM emission),
    and a baseline every other model must beat. Scaling matters because metres and
    degrees have different units. class_weight=None keeps probabilities calibrated.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000,
                                   solver="lbfgs", random_state=SEED)),
    ])


def lightgbm(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0):
    """Gradient-boosted trees (LightGBM): many small trees, each fixing the last ones' errors.

    Handles nonlinear rules such as "close AND same direction" without hand-made
    interactions, needs no scaling, is fast on ~1.2M rows, and records a real
    per-iteration validation loss (the only boosting loss curve we plot).
    """
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def get_models():
    """Dict of name -> (algorithm family, untrained estimator).

    Milestone 1: Logistic Regression + LightGBM. Then Random Forest, MLP and
    distance-weighted KNN are added here.
    """
    return {
        "logistic_regression": ("Linear / probabilistic", logistic_regression()),
        "lightgbm": ("Boosting", lightgbm()),
    }
