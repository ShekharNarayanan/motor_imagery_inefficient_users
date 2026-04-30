# ml utilities for the project
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from motor_imagery_inefficient_users import extract_features
import numpy as np


def csp_lda_cv_per_participant(
    X_epoched: np.ndarray,
    y: list,
    fs: int,
    csp_params: dict,
) -> list:
    """Evaluate FBCSP+LDA pipeline via stratified k-fold cross-validation.

    Args:
        X_epoched (np.ndarray): epoched EEG data, shape (n_trials, n_channels, n_times)
        y (np.ndarray): class labels, shape (n_trials,)
        fs (int): sampling frequency in Hz
        csp_params (dict): csp_params (dict): parameters for the filter used. contains the filter bands, number of components, filter category, order and type

    Returns:
        list: accuracy scores for each fold
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_epoched, y):
        X_train, X_test = X_epoched[train_idx], X_epoched[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_features, csps = extract_features.filter_bank_CSP(
            X_train,
            y_train,
            fs,
            training=True,
            csp_params=csp_params,
        )
        test_features, _ = extract_features.filter_bank_CSP(
            X_test,
            None,
            fs,
            training=False,
            all_csps=csps,
            csp_params=csp_params,
        )

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(train_features, y_train)
        scores.append(lda.score(test_features, y_test))

    return scores


def csp_lda_final_model(
    X_epoched: np.ndarray,
    y: list,
    fs: int,
    csp_params: dict,
):
    """
    Computes the csp weights and features that are ultimately used for classification in LDA.
    CSP weights are stored such that they can be used for computing features of the test data (data collected after calibration)

    Args:
        X_epoched (np.ndarray): epoched eeg data
        y (list): labels for each trial
        fs (int): sampling freq
        csp_params (dict): parameters for the filter used. contains the filter bands, number of components, filter category, order and type

    """

    train_features, csps = extract_features.filter_bank_CSP(
        X_epoched,
        y,
        fs,
        training=True,
        csp_params=csp_params,
    )
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(train_features, y)

    return csps, lda
