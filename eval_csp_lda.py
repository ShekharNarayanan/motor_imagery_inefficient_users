# evaluation script for the csp lda pipeline
import yaml
import pandas as pd
from motor_imagery_inefficient_users import io, preprocess, extract_features
import os
import joblib

# load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# load config params
test_data_path = config["data_test"]
fs = config["fs"]
task_start_time_s = config["task_start_time_s"]
task_end_time_s = config["task_end_time_s"]
num_trials = config["num_trials"]
num_chans = config["num_chans"]
ignore_participants = config["ignore_participants"]

# remove participants left out by the publication
all_pids = io.get_participant_ids(data_path=test_data_path)
pids = [
    pid for pid in all_pids if pid not in ignore_participants
]  # ignore the ones authors did not use


# loop for remaining participants
for pid in pids:
    print(f"Working with participant {pid} ")
    test_pid_files = io.get_test_data_for_participant(
        test_data_path=test_data_path, participant_id=pid
    )

    print(f"files present: {[str(f) for f in test_pid_files]}")

    # combine files for each participant
    merged_pid_data = pd.concat([pd.read_csv(f) for f in test_pid_files])

    print(merged_pid_data.shape)

    # get epoched data from the merged file
    num_trials_test = num_trials * len(test_pid_files)  # get the total number of trials
    X_test, y_test = preprocess.get_epoched_eeg_and_labels(
        eeg_df=merged_pid_data,
        num_chans=num_chans,
        num_trials=num_trials_test,
        task_start_time_s=task_start_time_s,
        task_end_time_s=task_end_time_s,
        fs=fs,
    )

    print(f"Shape of test data: X:{X_test.shape},y: {y_test.shape}")
    break


# load lda fit and csp weights for participant

# get prediction results

# save results
