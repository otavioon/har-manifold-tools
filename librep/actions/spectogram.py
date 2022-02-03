import numpy as np
from scipy import signal
from .action import Action
from functools import partial
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib

elementwise_mean = partial(np.mean, axis=0)

class ExtractSpectrogram(Action):
    def __init__(self, fs: float, nperseg: int, nooverlap: int):
        self.fs = fs
        self.nperseg = nperseg
        self.nooverlap = nooverlap
        self.res = None
        
    def run(self, input_data: np.ndarray, metadata: dict) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
        f, t, s = signal.spectrogram(input_data, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
        self.res = (f, t, s)
        return ((f, t, s), metadata)
    
    
class ExtractCombinedSpectrogram(Action):
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
        self.res = None
        
    def run(self, input_data: tuple, metadata: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        specs = []
        for i in input_data:
            f, t, s = signal.spectrogram(i, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
            specs.append((f, t, s))
        fs = self.frequency_combination_method(list(zip(*specs))[0])
        ts = self.time_combination_method(list(zip(*specs))[1])
        ss = self.spectrogram_combination_method(list(zip(*specs))[2])
        self.res = (fs, ts, ss)
        return ((fs, ts, ss), metadata)
    
    
class ExtractCombinedLogSpectrogram(Action):
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
        self.res = None
        
    def run(self, input_data: tuple, metadata: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        specs = []
        for i in input_data:
            f, t, s = signal.spectrogram(i, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
            specs.append((f, t, 10*np.log10(s)))
        fs = self.frequency_combination_method(list(zip(*specs))[0])
        ts = self.time_combination_method(list(zip(*specs))[1])
        ss = self.spectrogram_combination_method(list(zip(*specs))[2])
        self.res = (fs, ts, ss)
        return ((fs, ts, ss), metadata)
    
    
class PlotSpectrogram(Action):
    def __init__(self, figtitle: str = "", xlabel: str = "Time (s)", ylabel: str = "Freq (Hz)", vmin: float = 0.0, vmax: float = 1.0, figsize=None, shading='gouraud', output_path: str = None, display: bool = True, plot_axis: bool = True, cmap=None):
        self.figtitle = figtitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vmin = vmin
        self.vmax = vmax
        self.figsize = figsize if figsize else (12, 8)
        self.shading = shading
        self.output_path = output_path
        self.plot_axis = plot_axis
        self.display = display
        self.cmap=cmap
        self.fig = None
        self.ax = None

    def run(self, input_data: Tuple[np.ndarray, np.ndarray, np.ndarray], metadata: dict):
        fig = plt.figure(frameon=False, figsize=self.figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        self.fig = fig
        self.ax = ax
        fig.suptitle(self.figtitle)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        #matplotlib.colors.LogNorm(vmin=input_data[2].min(), vmax=input_data[2].max())

        x = ax.pcolormesh(input_data[1], input_data[0], input_data[2], shading=self.shading, vmin=self.vmin, vmax=self.vmax)
        #fig.colorbar(x)
        
        if not self.plot_axis:
            ax.set_axis_off()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for item in [fig, ax]:
                item.patch.set_visible(False)
            plt.box(False)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
        if self.output_path:
            plt.savefig(self.output_path, bbox_inches='tight', transparent=False, pad_inches=0, dpi=56)
        if self.display:
            plt.show()
        plt.close()
        return (input_data, metadata)

    
class PlotMultiSpectrogram(Action):
    def __init__(self, figtitle: str, 
                 nrows: int, 
                 ncols: int, 
                 row_names: List[str], 
                 col_names: List[str], 
                 xlabel: str, 
                 ylabel: str, 
                 vmin: float = 0.0, 
                 vmax: float = 3.0, 
                 figsize=None, 
                 shading='gouraud',
                 sharex=False,
                 sharey=False,
                 output_path: str = None):
        self.figtitle = figtitle
        self.nrows = nrows
        self.ncols = ncols
        self.row_names = row_names
        self.col_names = col_names
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vmin = vmin
        self.vmax = vmax
        self.figsize = figsize
        self.shading = shading
        self.sharex = sharex
        self.sharey = sharey
        self.output_path = output_path
        self.fig = None
        self.axs = None
    
    def run(self, input_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], metadata: dict):
        figsize = (self.ncols*3, self.nrows*2) if not self.figsize else self.figsize
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize, squeeze=False, sharex=self.sharex, sharey=self.sharey)    
        self.fig = fig
        self.axs = axs
        plt.subplots_adjust(top=0.80, hspace=0.9, wspace=0.2)
        fig.suptitle(self.figtitle, y=0.85, x=0.5, fontsize=18)
        for i in range(self.nrows):
            for j in range(self.ncols):
                data = input_data[i][j]
                if data is None:
                    continue
                axs_title = f"{self.row_names[i]} ({self.col_names[j]})"                
                axs[i,j].set_ylabel(self.ylabel)
                axs[i,j].set_xlabel(self.xlabel)
                axs[i,j].set_title(axs_title)
                axs[i,j].pcolormesh(data[1], data[0], data[2], shading=self.shading, vmin=self.vmin, vmax=self.vmax)
                
        if self.output_path:
            plt.savefig(self.output_path, bbox_inches='tight', transparent=False)
        plt.show()
        #plt.close()