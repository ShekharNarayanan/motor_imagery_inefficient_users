# ml utilities for the project
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from motor_imagery_inefficient_users import extract_features
import numpy as np



def csp_lda_cv_per_participant(X_epoched:np.ndarray, y:list, freq_bands:list, fs:int, csp_components:int) -> list:
    """Evaluate FBCSP+LDA pipeline via stratified k-fold cross-validation.

    Args:
        X_epoched (np.ndarray): Epoched EEG data, shape (n_trials, n_channels, n_times)
        y (np.ndarray): Class labels, shape (n_trials,)
        freq_bands (list): List of (low, high) tuples defining frequency bands
        fs (int): Sampling frequency in Hz
        csp_components (int): Number of CSP components per band

    Returns:
        list: Accuracy scores for each fold
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_epoched, y):
        X_train, X_test = X_epoched[train_idx], X_epoched[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_features, csps = extract_features.filter_bank_CSP(X_train, y_train, freq_bands, fs, csp_components=csp_components, training=True)
        test_features, _ = extract_features.filter_bank_CSP(X_test, None, freq_bands, fs, csp_components=csp_components, training=False, all_csps=csps)

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(train_features, y_train)
        scores.append(lda.score(test_features, y_test))

    return scores

def csp_lda_final_model(X_epoched:np.ndarray, y:list, freq_bands:list, fs:int, csp_components:int):
    """_summary_

    Args:
        X_epoched (np.ndarray): _description_
        y (list): _description_
        freq_bands (list): _description_
        fs (int): _description_
        csp_components (int): _description_
    """

    train_features, csps = extract_features.filter_bank_CSP(X_epoched, y, freq_bands, fs, csp_components=csp_components, training=True)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(train_features, y)

    return csps, lda


