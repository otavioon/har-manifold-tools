from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy import signal
from functools import partial

elementwise_mean = partial(np.mean, axis=0)

class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class SpectrogramCombiner:
    def __init__(self, fs: float, nperseg: int, nooverlap: int, 
                 spectrogram_combination_method=sum,
                 frequency_combination_method=elementwise_mean,
                 time_combination_method=elementwise_mean):
        self.fs = fs
        self.nperseg = nperseg
        self.nooverlap = nooverlap
        self.frequency_combination_method = frequency_combination_method
        self.time_combination_method = time_combination_method
        self.spectrogram_combination_method = spectrogram_combination_method
        
    
    def __call__(self, temporal_sample):
        specs = []
        for i in range(temporal_sample.shape[1]):
            f, t, s = signal.spectrogram(temporal_sample[:, i], fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
            specs.append((f, t, s))
        fs = self.frequency_combination_method(list(zip(*specs))[0])
        ts = self.time_combination_method(list(zip(*specs))[1])
        ss = self.spectrogram_combination_method(list(zip(*specs))[2])
        return fs, ts, ss