# preprocessing script for project
import pandas as pd
import numpy as np

def get_epoched_eeg_and_labels(X_continuous:pd.DataFrame, num_chans:int, num_trials:int, task_start_time_s:int, task_end_time_s:int, fs:int)-> np.ndarray:
    """_summary_

    Args:
        X_continuous (pd.DataFrame): _description_
        num_trials (int): _description_
        task_start_time_s (int): _description_
        task_end_time_s (int): _description_
        fs (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # prepare data and keep only relevant columns
    X_tmp = X_continuous.copy()
    y_tmp = np.array(X_continuous["class"])

    # keep only channel data
    X_tmp.drop(columns = ["TimeStamp","trial","class"], inplace =True)
    X_tmp = np.array(X_tmp)

    # compute trial related params
    task_start_time_samples = task_start_time_s * fs
    task_end_time_samples   = task_end_time_s * fs
    trial_duration_samples = int(task_end_time_samples - task_start_time_samples)

    # initialize epoched matrix
    X_epoched = np.zeros((num_trials, num_chans, trial_duration_samples))
    y = np.zeros((num_trials, trial_duration_samples))

    # prepare labels
    y_tmp[y_tmp == -1] = 0 # rename classes from 1 and -1 to 1 and 0
    

    for i_trial in range(num_trials):
        start = i_trial * trial_duration_samples
        end = start + trial_duration_samples
        X_epoched[i_trial,:,:] =  X_tmp[start:end,:].T
        y[i_trial,:] = y_tmp[start:end]

    # Check every row has all-same values
    assert (y == y[:, [0]]).all(), "Some rows have differing values"
    y = y[:,0] # take the 40 x 1 vector that contains the labels

    assert X_epoched.shape[0] == len(y), "dims of X_epoched and labels dont match"

    return X_epoched, y





