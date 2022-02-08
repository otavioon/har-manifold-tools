import functools
import warnings
import random
from typing import Generic, Iterator, Sequence, TypeVar, Optional, Callable

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
        
class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
        

class Subset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class SimpleDataset(Dataset):
    def __init__(self, values):
        self.values = values
        
    def __getitem__(self, index):
        return self.values[index]
    
    def __len__(self) -> int:
        return len(self.values)
    
    
class DataLoader(Generic[T_co]):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, 
                batch_transform: Optional[Callable] = None, max_workers: int = 0):
        assert batch_size > 0, "batch size must be grater than 0"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_transform = batch_transform
        self.max_workers = max_workers

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
        

class SimpleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, 
                    batch_transform: Optional[Callable] = None, max_workers: int = 0):
        super().__init__(   dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                            batch_transform=batch_transform, max_workers=max_workers)
        self.indices = list(range(len(dataset)))
        self.i = 0

    def __iter__(self):
        self.i=0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.max_workers == 0:
            if self.i >= len(self.dataset)-self.batch_size:
                raise StopIteration
            
            values = []
            for j in range(self.i, self.i+self.batch_size):
                values.append(self.dataset[self.indices[j]])
            if self.batch_transform is not None:
                values = self.batch_transform(values)

            self.i += self.batch_size
            return values
        else:
            # TODO multi workers
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)//self.batch_size