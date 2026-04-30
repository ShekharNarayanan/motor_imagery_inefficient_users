# input/output operations related to the project
import numpy as np
import pandas as pd
from pathlib import Path
import re

def get_calibration_data_for_participant(calibration_data_path:str, participant_id:int)->pd.DataFrame:
    """Get calibration data for participant. 

    Args:
        calibration_data_path (str): path for calibration data
        participant_id (int): id of the participant

    Returns:
        pd.DataFrame: data frame containing time-series eeg data for all channels and trial info.
    """

    pid_data = pd.read_csv(f"{calibration_data_path}/S{participant_id}_eeg_Calibration.csv")

    return pid_data

def get_test_data_for_participant(test_data_path:str, participant_id:int)->pd.DataFrame:
    """Get test data for participant. 

    Args:
        test_data_path (str): path for test data
        participant_id (int): id of the participant

    Returns:
        pd.DataFrame: data frame containing time-series eeg data for all channels and trial info.
    """
    pid_data = pd.read_csv(f"{test_data_path}/Subject{participant_id}_eeg_Calibration.csv")

    return pid_data

def get_participant_ids(data_path:str)->list[int]:
    """Gets all participant ids in a given folder. Accomodates the 'S{id}' and the 'Subject{id}' formats.

    Args:
        data_path (str): folder for which participant ids need to be found.

    Returns:
        np.ndarray: array of participant ids
    """
    pattern = r"(?:S|Subject)(\d+)_eeg.*\.csv"  # note the capturing group
    
    pids = []
    for f in Path(data_path).glob("*.csv"):
        m = re.match(pattern, f.name)
        if m:
            pid = int(m.group(1))  # extracted participant ID
            pids.append(pid)
    
    return sorted(pids)

