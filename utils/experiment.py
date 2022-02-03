import os
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import itertools
import time

from typing import List, Tuple, Dict, Optional
from functools import partial
import ucc.user_centric_coordinates as uc_coord

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy import signal
from scipy.fft import fftshift
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from utils.dataset import KuHarDataset

class Action:
    def __init__(self, action_title: str):
        self.title = action_title
    
    def prepare(self, input_data):
        return input_data
    
    def run(self, input_data):
        return input_data

    def describe_config(self) -> dict:
        return {}
    
    def save(self, expid: str, output_dir: str) -> bool:
        return True

class KuHarSelect(Action):
    def __init__(self, title: str, users: List[int] = None, 
                 activities: List[int] = None, 
                 columns: List[str]=None):
        super().__init__(title)
        self.users = users
        self.activities = activities
        self.columns = columns
       
    def run(self, input_data: KuHarDataset) -> pd.DataFrame:
        values = []
        for d in input_data.get_element_data_iterator(users=self.users, activities=self.activities):
            v = d.copy()
            values.append(v[self.columns])
        return pd.concat(values)
    
    def describe_config(self) -> dict:
        return {
            'users': self.users,
            'activities': self.activities,
            'columns': self.columns
        }

class DataFrameSelect(Action):
    def __init__(self, title: str, xs: List[str], reshape=None):
        super().__init__(title)
        self.xs = xs
        self.reshape = reshape
        
    def run(self, input_data: pd.DataFrame) -> np.ndarray:
        x = input_data[self.xs].values
        if self.reshape is not None:
            x = x.reshape(self.reshape)
        return x
    
    def describe_config(self) -> dict:
        return {
            'xs': self.xs,
            'reshape': self.reshape
        }
    
class DataFrameSplit(Action):
    def __init__(self, title: str, xs: List[str], ys: List[str]):
        super().__init__(title)
        self.xs = xs
        self.ys = ys
        
    def run(self, input_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs = input_data[self.xs].values
        ys = input_data[self.ys].values
        return xs, ys
    
    def describe_config(self) -> dict:
        return {
            'xs': self.xs,
            'ys': self.ys
        }
      
class ExtractSpectogram(Action):
    def __init__(self, title: str, fs: float, nperseg: int, nooverlap: int):
        super().__init__(title)
        self.fs = fs
        self.nperseg = nperseg
        self.nooverlap = nooverlap
        self.f = None
        self.t = None
        self.s = None
        
    def run(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, t, s = signal.spectrogram(input_data, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
        self.f = f
        self.t = t
        self.s = s
        return (f, t, s)
    
    def describe_config(self) -> dict:
        return {
            'Sample frequency': self.fs,
            'Segment lenght': self.nperseg,
            'Overlap': self.nooverlap
        }
    
    def save(self, expid: str, output_dir: str) -> bool:
        #fname = os.path.join(output_dir, f"{expid}.fft.fs-{self.fs}.nseg-{self.nperseg}.over-{self.nooverlap}")
        #np.savez_compressed(fname, f=self.f, t=self.t, s=self.s)
        return True
        
        
class GetSpectogramStatistics(Action):
    def __init__(self, title: str, action: str):
        super().__init__(title)
        self.action = action
        
    def run(self, input_data):
        f, t, s = input_data
        s_min = np.min(s)
        s_max = np.max(s)
        s_avg = np.mean(s)
        s_std = np.std(s)
        print(f"[{self.action}] Smin: {s_min}, Smax: {s_max}, Savg: {s_avg}, Sstd: {s_std}")
        return input_data
        
class PlotMultiSpectogram(Action):
    def __init__(self, title: str, figtitle: str, 
                 nrows: int, ncols: int, 
                 row_names: List[str], col_names: List[str], 
                 xlabel: str, ylabel: str, 
                 vmin: float = 0.0, vmax: float = 3.0, 
                 figsize=None, shading='gouraud'):
        super().__init__(title)
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
        self.fig = None
        self.axs = None
    
    def run(self, input_data: List[tuple]):
        fig, axs = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)
        #plt.rcParams['figure.constrained_layout.use'] = True
        self.fig = fig
        self.axs = axs
        fig.suptitle(self.figtitle)
        for i in range(self.nrows):
            for j in range(self.ncols):
                data = input_data[i][j]
                if data is None:
                    continue
                axs_title = f"{self.row_names[i]} ({self.col_names[j]})"
                axs[i*self.nrows+j].pcolormesh(data[1], data[0], data[2], shading=self.shading, vmin=self.vmin, vmax=self.vmax)
                axs[i*self.nrows+j].set_ylabel(self.ylabel)
                axs[i*self.nrows+j].set_xlabel(self.xlabel)
                axs[i*self.nrows+j].set_title(axs_title)
        plt.subplots_adjust(top=1.85)
        fig.tight_layout(pad=2.0)
        fig.show()
    
    def save(self, expid: str, output_dir: str) -> bool:
        path = os.path.join(output_dir, f"{expid}.png")
        plt.savefig(path, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {path}")
        return True
    
        
class TrainKNN(Action):
    def __init__(self, title: str, n_neighbors, test_size=0.2, random_state: int = 0):
        super().__init__(title)
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state
        
    def prepare(self, input_data: Tuple[np.ndarray, np.ndarray]):
        X_train, X_test, y_train, y_test = train_test_split(input_data[0], input_data[1], test_size=self.test_size)
        return X_train, X_test, y_train, y_test
    
    def run(self, input_data: Tuple[np.ndarray, np.ndarray]):
        neigh = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        neigh.fit(input_data[0], input_data[2])
        return neigh
    
    def describe_config(self) -> dict:
        return {
            'Neighbors': self.n_neighbors
        }
    
    
class ExperimentError(Exception):
    pass
    
class Experiment:
    exp_counter: int = 0
    
    def __init__(self, title: str, description: str, output_dir: str, actions: List[Action], expdesc: str = ''):
        Experiment.exp_counter += 1
        self.title = title
        self.expid = f"{expdesc}-{Experiment.exp_counter}"
        self.description = description
        self.output_dir = output_dir
        self.actions = actions
        self.result = None
        
    def run(self, input_data):
        r = input_data
        os.makedirs(self.output_dir, exist_ok=True)
        start = time.time()
        #print(f"*****Experiment {self.expid} STARTED******")
        for a in self.actions:
            #print(f" ---- Executing action {a.title} of experiment {self.title} ({self.expid})------ ")
            r = a.prepare(r)
            r = a.run(r)
            self.result = r
            a.save(self.expid, self.output_dir)
        #print(f"*****Experiment FINISHED (took {time.time()-start:.3f} seconds)******\n")
            
    def save(self, *args, **kwargs):
        # Must check this
        pickle.dump(self, output_dir)
    

class ExperimentManager:
    def __init__(self):
        pass