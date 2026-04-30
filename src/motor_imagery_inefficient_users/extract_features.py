# extract relevant features
from mne.decoding import CSP
from mne.filter import filter_data
import numpy as np
import logging
from mne.utils import set_log_level

set_log_level("WARNING")  # mutes MNE info messages
logging.getLogger("sklearn").setLevel(logging.WARNING)


def filter_bank_CSP(
    X: np.ndarray,
    y: np.ndarray,
    fs: int,
    training: bool = True,
    all_csps: list = None,
    csp_params:dict = None
) -> tuple[np.ndarray, ...]:
    """Performs filter bank Common Spatial Filtering for given eeg data and labels. 
       For each frequency band, the following is done:
       Computes the CSP fit to find 'w' and then the csp transform (w.t @ X) to find the log variance of the features.
       CSP fits computed during training are used for finding features of test data. Later used to fit LDA.

    Args:
        X (np.ndarray): epoched eeg data
        y (np.ndarray): labels
        bands (tuple[list]): a tuple containing lists of different frequency bands f_i. f_i = [low, high]
        fs (int): sampling frequency of eeg data
        csp_components (int): number of csp components to be computed.
        training (bool, optional): set to True if filter bank CSP is applied during training. Defaults to True.
        all_csps (list, optional): csp weights computed using .fit(). Defaults to None.
        csp_params (dict): parameters for the filter used. contains the filter bands, number of components, filter category, order and type

    Returns:
        tuple[np.ndarray, ...]: Concatenated features and CSP weights
    """
    # initialize output vars
    fitted_csps = []
    features = []

    # get all csp params
    bands = csp_params.get("bands",[])
    csp_components = csp_params.get("csp_components",4)
    method = csp_params.get("category",'iir')
    filter_order = csp_params.get("order",5)
    filter_type = csp_params.get("type")


    assert len(bands) != 0, 'Please enter freq bands for CSP in config file.'


    # loop over all freq bands
    for i, (low, high) in enumerate(bands):
        # filter eeg data
        X_filt = filter_data(
            data=X,
            sfreq=fs,
            l_freq=low,
            h_freq=high,
            method=method,
            iir_params={"order": filter_order, "ftype": filter_type},
        )
        # if features are being computed during training, fit the csp
        if training:
            csp = CSP(n_components=csp_components, log=True)
            csp.fit(X=X_filt, y=y)

        # if features are being computed for test data then use weights computed during training
        else:
            csp = all_csps[i]

        # append results
        fitted_csps.append(csp)
        features.append(
            csp.transform(X_filt)
        )  # transform gives a X.shape[0] * num_components

    return np.concatenate(features, axis=1), fitted_csps
