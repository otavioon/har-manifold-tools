from json import load
import time
from typing import Any, List
from unittest import loader
import uuid
from .dataset import DataLoader

class Experiment:
    def __init__(self, func, description: str) -> None:
        self.created_time = time.time()
        self.func = func
        self.description = description
        self.run_count = 0
        self.result = None

    def __call__(self, data: Any):
        self.run_count += 1
        self.result = self.func(data)
        return self.result

    def is_executed(self):
        return self.run_count > 0


class ExperimentRunner:
    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments
        self.i = 0

    def run(self, data):
        for exp in self.experiments:
            print(f"--- Running experiment: {exp.description} ---")
            exp(data)

        return self.experiments

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.experiments):
            raise StopIteration
        res = self.experiments[self.i]
        self.i += 1
        return res

