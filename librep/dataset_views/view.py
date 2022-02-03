from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class DatasetView(ABC):   
    @abstractmethod
    def to_canonical(self) -> Tuple[np.ndarray, pd.DataFrame]:
        raise NotImplementedError("Must be implemented in dervived classes")