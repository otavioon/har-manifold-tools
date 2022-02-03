from curses import meta
import os
from platform import release
from random import sample
from threading import activeCount
import numpy as np
import pandas as pd
import glob
from typing import List, Callable, Optional, Tuple, Dict, Any

from .dataset import MultiModalTemporalDataset, TemporalSample
from libhiaac.utils.file_ops import download_url, unzip_file
from libhiaac.utils.attribute_dict import AttributeDict

# Activity names and codes
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

activity_codes = {
    "Stand": 0,
    "Sit": 1,
    "Talk-sit": 2,
    "Talk-stand": 3,
    "Stand-sit": 4,
    "Lay": 5,
    "Lay-stand": 6,
    "Pick": 7,
    "Jump": 8,
    "Push-up": 9,
    "Sit-up": 10,
    "Walk": 11,
    "Walk-backwards": 12,
    "Walk-circle": 13,
    "Run": 14,
    "Stair-up": 15,
    "Stair-down": 16,
    "Table-tennis": 17
}


class RawKuHar(MultiModalTemporalDataset):
    # Version 5 KuHar Raw
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/d3126562-b795-4eba-8559-310a25859cc7/file_downloaded"

    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        transforms_target: Optional[Callable] = None,
        download: bool = False
    ):
        super().__init__(dataset_path=dataset_path, transforms=transforms,
            transforms_target=transforms_target, download=download
        )

        if self.download:
            self._download()
            
        self.metadata_df = self._read_metadata()
        
    def _download(self):
        # Create directories
        os.makedirs(self.dataset_path, exist_ok=True)
        fname = os.path.join(self.dataset_path, "dataset.zip") 
        if not os.path.exists(fname):
            print(f"Downloading dataset to '{fname}'")
            download_url(self.dataset_url, fname=fname)
        else:
            print(f"'{fname}' already exists and will not be downloaded again")
        print(f"Unziping dataset to {self.dataset_path}")
        unzip_file(filename=fname, destination=self.dataset_path)
        print(f"Removing {fname}")
        os.unlink(fname)
        print("Done!")

    def _read_metadata(self):
        # Let's list all CSV files in the directory
        files = glob.glob(os.path.join(self.dataset_path, "*", "*.csv"))

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
        column_dtypes = [
            ("class", np.int),
            ("cname", str),
            ("user", np.int),
            ("serial", np.int), 
            ("file", str)
        ]
        metadata_df = pd.DataFrame(users_relation, columns=[d[0] for d in column_dtypes])
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df
    
    def _read_csv_data(self, info) -> pd.DataFrame:
        # Default feature names from this dataset
        feature_dtypes = {
            "accel-start-time": np.float, 
            "accel-x": np.float,
            "accel-y": np.float, 
            "accel-z": np.float, 
            "gyro-start-time": np.float, 
            "gyro-x": np.float, 
            "gyro-y": np.float, 
            "gyro-z": np.float
        }
        
        with open(info['file'], 'r') as f:
            csv_matrix = pd.read_csv(f, names=list(feature_dtypes.keys()), dtype=feature_dtypes)
            #csv_matrix["User"] = info["User"]
            #csv_matrix["Action Code"] = info["Action Code"]
            #csv_matrix["Sequence"] = info["Sequence"]
            
            # Reordering to same format as all Ku-Har datasets
            csv_matrix = csv_matrix[[
                "accel-x", "accel-y", "accel-z", 
                "gyro-x", "gyro-y", "gyro-z", 
                "accel-start-time", "gyro-start-time"
            ]]
            csv_matrix["accel-end-time"] = csv_matrix["accel-start-time"]
            csv_matrix["gyro-end-time"] = csv_matrix["gyro-start-time"]
            csv_matrix["class"] = info["class"]
            csv_matrix["length"] = 1
            csv_matrix["serial"] = info["serial"]
            csv_matrix["index"] = range(len(csv_matrix))
            csv_matrix["user"] = info["user"]
            return csv_matrix
    
    def get_all_user_ids(self) -> List[int]:
        return np.sort(self.metadata_df["user"].unique()).tolist()
    
    def get_all_activity_ids(self) -> List[int]:
        return np.sort(self.metadata_df["class"].unique()).tolist()
    
    def get_data_iterator(self, users: List[int] = None, activities: List[int] = None, shuffle: bool = False) -> List[pd.DataFrame]:
        # Must select first
        if users is None:
            users = self.get_all_user_ids()
        if activities is None:
            activities = self.get_all_activity_ids()
            
        selecteds = self.metadata_df[
            (self.metadata_df["user"].isin(users)) & 
            (self.metadata_df["class"].isin(activities))
        ]
        
        # Shuffle data
        if shuffle:
            selecteds = selecteds.sample(frac=1)
        
        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self._read_csv_data(row)
            yield data
           
    def __getitem__(self, index: int) -> AttributeDict:
        row = self.metadata_df.iloc[index]
        data = self._read_csv_data(row)

        accel = data[["accel-x", "accel-y", "accel-z"]].values
        accel_times = data[["accel-start-time"]].values.T.squeeze()
        accel_sample = TemporalSample(timestamps=accel_times, points=accel)
        
        gyro = data[["gyro-x", "gyro-y", "gyro-z"]].values
        gyro_times = data[["gyro-start-time"]].values.T.squeeze()
        gyro_sample = TemporalSample(timestamps=gyro_times, points=gyro)

        metadata = {
            "class": int(data["class"].unique().tolist()[0]),
            "user": int(data["user"].unique().tolist()[0]),
            "serial": int(data["serial"].unique().tolist()[0])
        }
        
        if self.transforms is not None:
            accel_sample = self.transforms(accel_sample)
            gyro_sample = self.transforms(gyro_sample)
                
        if self.transforms_target is not None:
            metadata = self.transforms_target(metadata)
        
        return AttributeDict({
            "accelerometer": accel_sample,
            "gyroscope": gyro_sample,
            "meta": metadata
        })

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __str__(self):
        return f"KuHar Dataset at: '{self.dataset_path}' ({len(self.metadata_df)} files, {len(self.get_all_user_ids())} users and {len(self.get_all_activity_ids())} activities)"

    def __repr__(self) -> str:
        return f"Kuhar Dataset at '{self.dataset_path}'"
    

class TrimmedKuHar(RawKuHar):
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded"

    
class TimeDomainSubsamplesKuHar(MultiModalTemporalDataset):
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/f2b31500-b2a4-48e6-be0b-ebf621c18d42/file_downloaded"
    
    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        transforms_target: Optional[Callable] = None,
        download: bool = False,
        dataset_csv_path: str = "KU-HAR_time_domain_subsamples_20750x300.csv",
        sample_size: int = 300
    ):
        super().__init__(dataset_path=dataset_path, transforms=transforms,
            transforms_target=transforms_target, download=download
        )
        self.dataset_csv_path = dataset_csv_path

        if self.download:
            self._download()

        self.sample_size = sample_size
        self.values = self.read_values()

    def _download(self):
        # Create directories
        os.makedirs(self.dataset_path, exist_ok=True)
        fname = os.path.join(self.dataset_path, "dataset.zip") 
        if not os.path.exists(fname):
            print(f"Downloading dataset to '{fname}'")
            download_url(self.dataset_url, fname=fname)
        else:
            print(f"'{fname}' already exists and will not be downloaded again")
        print(f"Unziping dataset to {self.dataset_path}")
        unzip_file(filename=fname, destination=self.dataset_path)
        print(f"Removing {fname}")
        os.unlink(fname)
        print("Done!")

    def read_values(self):
        dataset_path = os.path.join(self.dataset_path, self.dataset_csv_path)
        return pd.read_csv(dataset_path, header=None)

    def __getitem__(self, index: int) -> AttributeDict:
        timestamps = np.arange(0, 0.01*self.sample_size, 0.01)
        # Read accelerometer data
        accel = self.values.iloc[index, 0:self.sample_size*3].values
        accel = np.stack([accel[0:self.sample_size], accel[self.sample_size:self.sample_size*2], accel[self.sample_size*2:self.sample_size*3]], axis=1)
        accel_sample = TemporalSample(timestamps=timestamps, points=accel)
        # Read gyroscope data
        gyro = self.values.iloc[index, self.sample_size*3:self.sample_size*6].values
        gyro = np.stack([gyro[0:self.sample_size], gyro[self.sample_size:self.sample_size*2], gyro[self.sample_size*2:self.sample_size*3]], axis=1)
        gyro_sample = TemporalSample(timestamps=timestamps, points=gyro)
        # Read metadata
        metadata = self.values.iloc[index, self.sample_size*6:self.sample_size*6+4].values.tolist()
        metadata = {
            "class": int(metadata[0]),
            "serial": int(metadata[2]),
            "user": "unknown"
        }

        if self.transforms is not None:
            accel_sample = self.transforms(accel_sample)
            gyro_sample = self.transforms(gyro_sample)
                
        if self.transforms_target is not None:
            metadata = self.transforms_target(metadata)

        return AttributeDict({
            "accelerometer": accel_sample,
            "gyroscope": gyro_sample,
            "meta": metadata
        })

    def __len__(self) -> int:
        return len(self.values)


    def __str__(self):
        return f"KuHar Dataset at: '{self.dataset_path}'"

    def __repr__(self) -> str:
        return f"Kuhar Dataset at '{self.dataset_path}'"
    

class BalancedTimeDomainKuHar(MultiModalTemporalDataset):
    dataset_url = {
        "train": "https://raw.githubusercontent.com/otavioon/har-datasets/master/dataset_views/KuHar/train_val_test_300samples/1638561602/train.csv",
        "validation": "https://raw.githubusercontent.com/otavioon/har-datasets/master/dataset_views/KuHar/train_val_test_300samples/1638561602/val.csv",
        "test": "https://raw.githubusercontent.com/otavioon/har-datasets/master/dataset_views/KuHar/train_val_test_300samples/1638561602/test.csv"
    }

    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        transforms_target: Optional[Callable] = None,
        download: bool = False,
        mode: str = "train",
        sample_size: int = 300
    ):
        super().__init__(dataset_path=dataset_path, transforms=transforms,
            transforms_target=transforms_target, download=download
        )
        
        self.mode = mode
        if mode == "train":
            self.dataset_csv_path = "train.csv"
        elif mode == "validation":
            self.dataset_csv_path = "val.csv"
        elif mode == "test":
            self.dataset_csv_path = "test.csv"
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be: 'train', 'validation' or 'test'")

        if self.download:
            self._download()

        self.sample_size = sample_size
        self.values = self.read_values()

    def _download(self):
        # Create directories
        os.makedirs(self.dataset_path, exist_ok=True)
        fname = os.path.join(self.dataset_path, self.dataset_csv_path) 
        if not os.path.exists(fname):
            print(f"Downloading dataset to '{fname}'")
            download_url(self.dataset_url[self.mode], fname=fname)
        else:
            # TODO check for checksum
            print(f"'{fname}' already exists and will not be downloaded again")
        print("Done!")
        return f"Kuhar Dataset at '{self.dataset_path}'"

    def read_values(self):
        dataset_path = os.path.join(self.dataset_path, self.dataset_csv_path)
        return pd.read_csv(dataset_path, header=0)

    def __getitem__(self, index: int) -> AttributeDict:
        # Read metadata
        metadata = self.values.iloc[index, self.sample_size*6:self.sample_size*6+9].values.tolist()

        # Read accelerometer data
        accel = self.values.iloc[index, 0:self.sample_size*3].values
        accel = np.stack([accel[0:self.sample_size], accel[self.sample_size:self.sample_size*2], accel[self.sample_size*2:self.sample_size*3]], axis=1)
        accel_timestamps = np.arange(float(metadata[0]), float(metadata[2]), (float(metadata[2])-float(metadata[0]))/self.sample_size)
        accel_sample = TemporalSample(timestamps=accel_timestamps, points=accel)
        # Read gyroscope data
        gyro = self.values.iloc[index, self.sample_size*3:self.sample_size*6].values
        gyro = np.stack([gyro[0:self.sample_size], gyro[self.sample_size:self.sample_size*2], gyro[self.sample_size*2:self.sample_size*3]], axis=1)
        gyro_timestamps = np.arange(float(metadata[1]), float(metadata[3]), (float(metadata[3])-float(metadata[1]))/self.sample_size)
        gyro_sample = TemporalSample(timestamps=gyro_timestamps, points=gyro)

        # Create metadata
        metadata = {
            "class": int(metadata[4]),
            "serial": int(metadata[6]),
            "user": int(metadata[8])
        }

        if self.transforms is not None:
            accel_sample = self.transforms(accel_sample)
            gyro_sample = self.transforms(gyro_sample)
                
        if self.transforms_target is not None:
            metadata = self.transforms_target(metadata)

        return AttributeDict({
            "accelerometer": accel_sample,
            "gyroscope": gyro_sample,
            "meta": metadata
        })

    def __len__(self) -> int:
        return len(self.values)


    def __str__(self):
        return f"KuHar Dataset at: '{self.dataset_path}'"

    def __repr__(self) -> str:
        return f"Kuhar Dataset at '{self.dataset_path}'"

