"""Model zoo: the 5+ classifiers we compare. OWNER: Kushal.

Every model must support predict_proba. Scalers live inside a sklearn Pipeline so they
are fitted on training folds only. config.SEED is used for every random_state.

The task for every model: given one (GPS point, candidate arc) row of features,
output P(this candidate is the arc the car was really on).
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def xgboost(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0):
    """XGBoost: gradient-boosted trees grown level by level (depth-limited), second-order updates.

    Same boosting family as LightGBM but a different tree-growth rule (max_depth instead of
    num_leaves) and built-in L2 regularisation, so it is a fair second boosting model. Needs
    no scaling; the eval_set in train_models.py gives a real per-iteration validation loss.
    tree_method="hist" keeps it fast on ~1.5M rows.
    """
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=min_child_weight, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, tree_method="hist",
        eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0)


def random_forest(n_estimators=150, max_depth=14, min_samples_leaf=20, max_features="sqrt",
                  max_samples=0.5):
    """Random Forest: many deep-ish trees on random row/feature subsets, probabilities averaged.

    The bagging (Ensemble) counterpart to LightGBM's boosting. Depth, leaf size and
    max_samples are limited so 150 trees fit in laptop RAM and cannot memorise single
    trips; n_jobs=-1 uses all cores.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, max_samples=max_samples, n_jobs=-1, random_state=SEED)


def mlp(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3):
    """Small neural network (multi-layer perceptron) on standardised features.

    Neural-network family. early_stopping holds out 10% of the training rows and stops
    when that score stops improving, so loss_curve_ / validation_scores_ are real
    per-epoch histories (the only NN curve we plot). Note the inner hold-out is random
    rows, used only to choose the stopping epoch; the outer CV split is still by trip.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                              learning_rate_init=learning_rate_init, batch_size=512,
                              max_iter=60, early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=5, random_state=SEED)),
    ])


def knn_distance_weighted(n_neighbors=25):
    """K-nearest-neighbours in scaled feature space, closer neighbours vote more.

    Instance-based family: no training, just memory, so prediction is slow. It is trained
    on a fixed point-level subsample (see TRAIN_SUBSAMPLE) to keep fit/predict affordable.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)),
    ])


def naive_bayes(var_smoothing=1e-9):
    """Gaussian Naive Bayes: a probabilistic baseline that treats features as independent.

    For each class (arc / not arc) it learns a mean and variance per feature, then
    multiplies the per-feature likelihoods ("naive" = independence assumption). Our features
    are strongly correlated (distance to the arc vs. distance rank), so its probabilities
    are over-confident; it is included to show what that assumption costs. Scaling does not
    change the Gaussian fit, but var_smoothing is relative to the largest feature variance,
    so scaling makes that stabiliser behave the same for every feature.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", GaussianNB(var_smoothing=var_smoothing)),
    ])


# Models that train on a fixed random subsample of GPS *points* (all candidate rows of a
# chosen point are kept together) so the run stays affordable. Value = max training points
# per fit. Chosen by compute budget, not by looking at results. Validation is never sampled.
TRAIN_SUBSAMPLE = {"knn_distance_weighted": 40_000, "mlp": 100_000}


def get_models():
    """Dict of name -> (algorithm family, untrained estimator): the compared models."""
    return {
        "logistic_regression": ("Linear / probabilistic", logistic_regression()),
        "random_forest": ("Ensemble (bagging)", random_forest()),
        "lightgbm": ("Boosting", lightgbm()),
        "xgboost": ("Boosting", xgboost()),
        "mlp": ("Neural network", mlp()),
        "knn_distance_weighted": ("Instance-based", knn_distance_weighted()),
        "naive_bayes": ("Probabilistic (generative)", naive_bayes()),
    }


FACTORIES = {
    "logistic_regression": logistic_regression, "random_forest": random_forest,
    "lightgbm": lightgbm, "xgboost": xgboost, "mlp": mlp, "knn_distance_weighted": knn_distance_weighted,
    "naive_bayes": naive_bayes,
}


def build(name, params=None):
    """Untrained estimator `name` with hyperparameters `params` (dict, e.g. Optuna's best).

    Used by bayes_opt.py (every trial), export_model.py and train_models.py --final so the
    tuned model is rebuilt from the saved JSON rather than copied by hand.
    """
    return FACTORIES[name](**(params or {}))
