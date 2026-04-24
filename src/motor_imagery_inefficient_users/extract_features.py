# extract relevant features
from mne.decoding import CSP
from mne.filter import filter_data
import numpy as np
import logging
from mne.utils import set_log_level

set_log_level('WARNING')  # mutes MNE info messages
logging.getLogger('sklearn').setLevel(logging.WARNING)


def filter_bank_CSP(X, y, bands, fs, csp_components, training=True,all_csps=None) -> tuple[np.ndarray, ...]:
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        bands (_type_): _description_
        fs (_type_): _description_
        csp_components (_type_): _description_
        training (bool, optional): _description_. Defaults to True.
        all_csps (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple[np.ndarray, ...]: _description_
    """
    # initialize output vars
    fitted_csps = []
    features = []

    # loop over all freq bands
    for i, (low,high) in enumerate(bands):
        # filter eeg data
        X_filt = filter_data(
            data=X,
            sfreq=fs,
            l_freq=low,
            h_freq=high,
            method='iir',
            iir_params={'order': 5, 'ftype': 'butter'}
        )
        # if features are being computed during training, fit the csp 
        if training:
            csp = CSP(n_components=csp_components, log=True)
            csp.fit(X=X_filt,y=y) 

        # if features are being computed for test data then use weights computed during training
        else:
            csp = all_csps[i]

        # append results
        fitted_csps.append(csp)
        features.append(csp.transform(X_filt)) # transform gives a X.shape[0] * num_components

        
    return np.concatenate(features,axis=1), fitted_csps

