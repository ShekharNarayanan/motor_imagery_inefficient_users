# script used for training the csp + lda model on each participant
import yaml
import numpy as np
from motor_imagery_inefficient_users import io, preprocess, ml_utils
import os
import joblib

# load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# load config params
calibration_data_path   = config["data_calibration"]
fs                      = config["fs"]
task_start_time_s       = config["task_start_time_s"]
task_end_time_s         = config["task_end_time_s"]
num_trials              = config["num_trials"]
num_chans               = config["num_chans"]
ignore_participants     = config["ignore_participants"]
filter_bank_bands       = config["filter_bank_bands"]
num_csp_components      = config["num_csp_components"]

# get all participant ids from the calibration data folder
all_pids = io.get_participant_ids(data_path=calibration_data_path)
pids     = [pid for pid in all_pids if pid not in ignore_participants] # ignore the ones authors did not use

# loop over all participants
for pid in pids:
    print("#" * 100)
    print(f"Starting process for participant {pid}")

    # load pd.DataFrame file containing everything
    pid_eeg_df = io.get_calibration_data_for_participant(
        calibration_data_path=calibration_data_path, participant_id=pid
    )

    # get epoched eeg data and corrected labels from the dataframe
    X_epoched, y = preprocess.get_epoched_eeg_and_labels(
        eeg_df=pid_eeg_df,
        num_chans=num_chans,
        num_trials=num_trials,
        task_start_time_s=task_start_time_s,
        task_end_time_s=task_end_time_s,
        fs=fs,
    )
    print("Epoching finished")
    # filter bank CSP + LDA using kfold cross validation: this gives a training performance indication per participant (internal use)
    cv_scores_pid = ml_utils.csp_lda_cv_per_participant(X_epoched=X_epoched,
                                                     y=y,
                                                     freq_bands=filter_bank_bands,
                                                     fs=fs,
                                                     csp_components=num_csp_components)
    print("Computed cross validation scores")
    print(f"CV mean: {np.mean(cv_scores_pid):.2f} ± {np.std(cv_scores_pid):.2f}")
    
    # compute filter weights and lda performance for all trials (this will be used on test data)
    csp_filter_weights, lda = ml_utils.csp_lda_final_model(X_epoched=X_epoched,y=y,freq_bands=filter_bank_bands,fs=fs,csp_components=num_csp_components)
    
    # save cross validation scores, weights and fitted lda for participant
    result_dir = "csp_lda_models"
    os.makedirs(result_dir,exist_ok=True)
    joblib.dump({'cross_val_scores': cv_scores_pid,'csp_weights': csp_filter_weights, 'lda': lda}, f'{result_dir}/csp_lda_model_{pid}.joblib')

    print("models and weights stored")
    print("Done....")


