import os
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import itertools

from typing import List

class Dataset(object):
    pass

class HARDataset(Dataset):
    def get_all_user_ids(self) -> List[int]:
        pass
    
    def get_all_activity_ids(self) -> List[int]:
        pass
    
    def get_element_data(self, users: List[int] = None, activities: List[int] = None, batch_size: int = 1, shuffle: bool = True) -> List[pd.DataFrame]:
        pass
    
    def create_time_series(self, data: pd.DataFrame, window: int = 30, overlap: int = 10) -> pd.DataFrame:
        pass
    

class KuHarDataset(HARDataset):
    # Default feature names from this dataset
    feature_names = ["accel-time", "accel-x", "accel-y", "accel-z", "gyro-time", "gyro-x", "gyro-y", "gyro-z"]
    
    # Activity names
    activity_names = {
        0: "Stand",
        1: "Sit",
        2: "Talk-sit",
        3: "Talk-stand",
        4: "Stand-sit",
        5: "Lay",
        6: "Lay-stand",
        7: "Pick",
        8: "Jump",
        9: "Push-up",
        10: "Sit-up",
        11: "Walk",
        12: "Walk-backwards",
        13: "Walk-circle",
        14: "Run",
        15: "Stair-up",
        16: "Stair-down",
        17: "Table-tennis"
    }
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.metadata_df = self._read_metadata()
        
    def _read_metadata(self):
        # Let's list all CSV files in the directory
        files = glob.glob(os.path.join(self.dataset_dir, "*", "*.csv"))
        print(f"There are {len(files)} files in total")        

        # And create a relation of each user, activity and CSV file
        users_relation = []
        for f in files:
            # Split the path into a list
            dirs = f.split(os.sep)
            # Pick activity name (folder name, e.g.: 5.Lay)
            activity_name = dirs[-2]
            # Pick CSV file name (e.g.: 1052_F_1.csv)
            csv_file = dirs[-1]
            # Split activity number and name (e.g.: [5, 'Lay'])
            act_no, act_name = activity_name.split(".")
            act_no = int(act_no)
            # Split user code, act type and sequence number (e.g.: [1055, 'G', 1])
            csv_splitted = csv_file.split("_")
            user = int(csv_splitted[0])
            sequence = '_'.join(csv_splitted[2:])
            # Remove the .csv from sequence
            sequence = sequence[:-4]
            # Generate a tuple with the information and append to the relation's list
            users_relation.append((act_no, act_name, user, sequence, f))

        # Create a dataframe with all meta information
        metadata_df = pd.DataFrame(users_relation, columns=["Action Code", "Action Name", "User", "Sequence", "File"])
        return metadata_df
    
    def _read_csv_data(self, info) -> pd.DataFrame:
        with open(info['File'], 'r') as f:
            csv_matrix = pd.read_csv(f, names=self.feature_names)
            csv_matrix["User"] = info["User"]
            csv_matrix["Action Code"] = info["Action Code"]
            csv_matrix["Sequence"] = info["Sequence"]
            return csv_matrix
    
    def get_all_user_ids(self) -> List[int]:
        return np.sort(self.metadata_df["User"].unique()).tolist()
    
    def get_all_activity_ids(self) -> List[int]:
        return np.sort(self.metadata_df["Action Code"].unique()).tolist()
    
    def get_element_data_iterator(self, users: List[int] = None, activities: List[int] = None, shuffle: bool = True) -> List[pd.DataFrame]:
        # Must select first
        if users is None:
            users = self.get_all_user_ids()
        if activities is None:
            activities = self.get_all_activity_ids()
            
        # TODO: maybe verify ranges of users and activities?
        selecteds = self.metadata_df[
            (self.metadata_df["User"].isin(users)) & 
            (self.metadata_df["Action Code"].isin(activities))
        ]
        
        # Shuffle data
        if shuffle:
            selecteds = selecteds.sample(frac=1)
        
        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self._read_csv_data(row)
            yield data
    
    def create_time_series(self, data: pd.DataFrame, window: int = None, overlap: int = 0) -> pd.DataFrame:
        if window is None:
            window = data.shape[0]
            
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap value must be in range [0, window)")
        
        values = []
        column_names = []
        accel_time_idx = data.columns.get_loc("accel-time")
        gyro_time_idx = data.columns.get_loc("gyro-time")
        to_drop_columns = ["accel-time", "gyro-time"]
        last_columns = ["accl-start", "accl-end", "gyro-start", "gyro-end"]
        
        for i in range(0, data.shape[0]-window, window-overlap):
            window_df = data[i:i+window]
            window_df = window_df[self.feature_names]
            accl_start_time  = window_df.iat[0, accel_time_idx] 
            accl_end_time = window_df.iat[-1, accel_time_idx]
            gyro_start_time = window_df.iat[0, gyro_time_idx] 
            gyro_end_time = window_df.iat[-1, gyro_time_idx]
            window_df = window_df.drop(to_drop_columns, axis=1) 
            window_values = window_df.unstack().values
            temp = np.concatenate(
                (window_values, [accl_start_time, accl_end_time, gyro_start_time, gyro_end_time])
            )
            values.append(temp)
        
        for feat in self.feature_names:
            if feat in to_drop_columns:
                continue
            names = [f"{feat}-{i}" for i in range(window)]
            column_names += names
            
        column_names += last_columns
        return pd.DataFrame(values, columns=column_names)