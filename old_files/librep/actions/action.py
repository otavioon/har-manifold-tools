from abc import ABC, abstractmethod
from typing import Any
from tqdm.contrib.concurrent import thread_map

class Action(ABC):   
    @abstractmethod
    def run(self, input_data: Any, metadata: dict):
        raise NotImplementedError("Must be implemented in dervived classes")
