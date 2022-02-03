import functools
import warnings
from typing import Generic, Iterator, Sequence, TypeVar

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
        
class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
       
class Subset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
