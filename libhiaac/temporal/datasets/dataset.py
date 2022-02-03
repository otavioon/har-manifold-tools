from ctypes import pointer
from typing import Tuple, Callable, Optional, Dict, Sequence, Any, List

import numpy as np
from libhiaac.utils.dataset import Dataset
from libhiaac.utils.attribute_dict import AttributeDict


class TemporalSample:
    def __init__(self, timestamps: np.ndarray, points: np.ndarray):
        assert len(timestamps.shape) == 1, "Timestamps must have shape of 1"
        assert points.shape[0] == timestamps.shape[0], \
            "Timestamps and points have the same length in first dimension"
        self.timestamps = timestamps
        self.points = points
        self.shape = self.points.shape    

    def rate(self) -> Tuple[float, float]:
        times_diff = np.diff(self.timestamps)
        return np.average(times_diff), np.std(times_diff)

    def flatten(self, axis=1) -> np.ndarray:
        return np.stack(self.points, axis=axis).ravel()

    def __str__(self) -> str:
        avg, stdev = self.rate()
        return f"no. points={self.timestamps.shape[0]}, rate={avg:.4f} (+-{stdev:.4f})"

    def __repr__(self) -> str:
        return f"Temporal Sample: {str(self)}"

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self) -> int:
        return len(self.timestamps)



class TemporalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        transforms_target: Optional[Callable] = None,
        download: bool = False
    ):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.transforms_target = transforms_target
        self.download = download
    
    def __getitem__(self, index: int) -> Tuple[TemporalSample, AttributeDict]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class MultiModalTemporalDataset(TemporalDataset):
    def __getitem__(self, index: int) -> AttributeDict:
        raise NotImplementedError