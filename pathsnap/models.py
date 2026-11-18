"""Model zoo: the 5+ classifiers we compare. OWNER: Kushal.

Every model must support predict_proba. Put scalers inside a sklearn Pipeline so they
are fitted on training folds only. Use config.SEED for every random_state.
"""


def get_models():
    """Dict of name -> (algorithm family, untrained estimator).

    Planned: Logistic Regression, Random Forest, LightGBM, MLP, distance-weighted KNN.
    """
    raise NotImplementedError("TODO(Kushal)")
