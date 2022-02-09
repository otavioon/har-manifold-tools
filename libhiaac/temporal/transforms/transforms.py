from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy import signal
from functools import partial
import matplotlib.pyplot as plt

elementwise_mean = partial(np.mean, axis=0)


class Composer:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
class MultiComposer:
    def __init__(self, composers: List[Composer], combiner_method: Callable = None):
        self.composers = composers
        self.combiner_method = combiner_method
        
    def __call__(self, data):
        datas = []
        for composer in self.composers:
            datas.append(composer(data))
            
        if self.combiner_method:
            return self.combiner_method(datas)
        else:
            return datas
        

class SpectrogramExtractor:
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
        if len(temporal_sample.shape) == 1:
            return signal.spectrogram(temporal_sample, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
        else:
            specs = []
            for i in range(temporal_sample.shape[-1]):
                f, t, s = signal.spectrogram(temporal_sample[:, i], fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
                specs.append((f, t, s))
            f = self.frequency_combination_method(list(zip(*specs))[0])
            t = self.time_combination_method(list(zip(*specs))[1])
            s = self.spectrogram_combination_method(list(zip(*specs))[2])
            return f, t, s
    
    
class SpectrogramColorMesh:
    def __init__(self, figsize: tuple = (1, 1), shading: str = "gouraud",
                 vmin: float = 0.0, vmax: float = 1.0):
        self.figsize = figsize
        self.shading = shading
        self.vmin = vmin
        self.vmax = vmax
        
    # tuple where 0-> fs, 1->time, 2->signal
    def __call__(self, value: tuple):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.pcolormesh(value[1], value[0], value[2], shading=self.shading, 
                      vmin=self.vmin, vmax=self.vmax)
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return data