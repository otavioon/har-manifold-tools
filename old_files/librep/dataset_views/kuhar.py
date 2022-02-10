from typing import List, Tuple
from collections import defaultdict
from tqdm.contrib.concurrent import thread_map
import pandas as pd
import tqdm
import numpy as np
from .view import DatasetView
from ..datasets.kuhar import RawKuHarDataset


class KuHarView(DatasetView):
    def __init__(self, kuhar: RawKuHarDataset, 
                 features: Tuple[str]=("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"), 
                 metadata: Tuple[str]=("class", "user", "serial", "index"),
                 users: List[int] = None, 
                 activities: List[int] = None):
        self.features = features
        self.metadata = metadata
        self.kuhar = kuhar
        self.users = users
        self.activities = activities
        
    def to_canonical(self) -> Tuple[np.ndarray, pd.DataFrame]:
        features = list(self.features)
        meta = list(self.metadata)

        dataframes = list(self.kuhar.get_data_iterator(users=self.users, activities=self.activities))
        m = max([d.shape[0] for d in dataframes])
        arr = np.full((len(dataframes), len(features), m), fill_value=np.nan)
        metas = []

        for i, d in enumerate(dataframes):
            for j, c in enumerate(features):
                temp = d[c].to_numpy()
                arr[i, j, :temp.shape[0]] = temp
            metas.append(d[meta].iloc[0].tolist())

        metas = pd.DataFrame(metas, columns=meta)
        return arr, metas
    
    
    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        return f"KuHarView (no. users={users}, no. activities={activities})"
    
class TimeSeriesKuHarView:
    def __init__(self, kuhar: RawKuHarDataset, 
                 window: int, 
                 overlap: int = 0,
                 features: Tuple[str]=("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"), 
                 metadata: Tuple[str]=("class", "user", "serial", "index"),
                 users: List[int] = None, 
                 activities: List[int] = None):
        self.kuhar = kuhar
        self.features = features
        self.metadata = metadata
        self.window = window
        self.overlap = overlap
        self.users = users
        self.activities = activities
        
        if window is None or window < 1:
            raise ValueError("Window value must be a positive integer")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap value must be in range [0, window)") 
        
    def _create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.window
        overlap = self.overlap

        values = []
        column_names = []
        selected_features = [
            "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"
        ]
        
        for i in range(0, data.shape[0], window-overlap):
            window_df = data[i:i+window]
            print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window 
            window_values = window_df[selected_features].unstack().to_numpy()
            acc_time = window_df["accel-start-time"].iloc[0], window_df["accel-end-time"].iloc[-1]
            gyro_time = window_df["gyro-start-time"].iloc[0], window_df["gyro-end-time"].iloc[-1]
            act_class = window_df["class"].iloc[0]
            length = window
            serial = window_df["serial"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate(
                (window_values, [
                    acc_time[0], gyro_time[0], acc_time[1], gyro_time[1],
                    act_class, length, serial, start_idx, act_user
                ])
            )
            values.append(temp)

        # Name the cows    
        column_names = [f"{feat}-{i}" for feat in selected_features for i in range(window)]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()
            
        # Hack to maintain types
        for c in ["class", "length", "serial", "index", "user"]:
            df[c] = df[c].astype(np.int)
        print(f"Extracted subsamples: {len(df)}")   
        return df
        
    def to_canonical(self) -> Tuple[np.ndarray, pd.DataFrame]:
        it = self.kuhar.get_data_iterator(users=self.users, activities=self.activities)
        dataframes = thread_map(self._create_time_series, list(it), desc=f"Generating time series with {self.window} samples", max_workers=1)
        dataframes = pd.concat(dataframes)
        
        feature_prefixes = list(self.features)
        meta = list(self.metadata)
        features_list = [[c for c in dataframes.columns if c.startswith(f)] for f in feature_prefixes]

        arr = np.full((len(dataframes), len(features_list), self.window), fill_value=np.nan)
        metas = []
        
        for i in tqdm.tqdm(range(len(dataframes)), desc="Generating canonical data"):
            for j, fl in enumerate(features_list):
                temp = dataframes[fl].iloc[i].to_numpy()
                arr[i, j, :temp.shape[0]] = temp
            metas.append(dataframes[meta].iloc[i].tolist())

        metas = pd.DataFrame(metas, columns=meta)
        return arr, metas

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        window = 'whole data' if self.window is None else self.window
        overlap = self.overlap
        return f"KuHarTimeSeriesView (window={window}, overlap={overlap}, no. users={users}, no. activities={activities})"



