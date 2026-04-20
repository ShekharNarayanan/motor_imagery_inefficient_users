# input/output operations related to the project
import numpy as np
import pandas as pd
from pathlib import Path
import re

def get_calibration_data_for_participant(calibration_data_path:str, participant_id:int)->pd.DataFrame:
    """Get calibration data for participant. 

    Args:
        calibration_data_path (str): _description_
        participant_id (int): _description_

    Returns:
        pd.DataFrame: _description_
    """

    pid_data = pd.read_csv(f"{calibration_data_path}/S{participant_id}_eeg_Calibration.csv")

    return pid_data

def get_test_data_for_participant(test_data_path:str, participant_id:int)->pd.DataFrame:
    """_summary_

    Args:
        test_data_path (str): _description_
        particpant_id (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    pid_data = pd.read_csv(f"{test_data_path}/Subject{participant_id}_eeg_Calibration.csv")

    return pid_data

def get_participant_ids(data_path:str)->list[int]:
    """_summary_

    Args:
        data_path (str): _description_

    Returns:
        np.ndarray: _description_
    """
    pattern = r"(?:S|Subject)(\d+)_eeg.*\.csv"  # note the capturing group
    
    pids = []
    for f in Path(data_path).glob("*.csv"):
        m = re.match(pattern, f.name)
        if m:
            pid = int(m.group(1))  # extracted participant ID
            pids.append(pid)
    
    return pids

