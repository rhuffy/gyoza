from typing import Any
import pickle


class ProgramAnalyzer:
    def __init__(self) -> None:
        self.cache = {}

    def __call__(self, program: str) -> Any:
        return self.cache.get(program, [0.0 for _ in range(10)])

    def load(self, pickled_stat_cache):
        with open(pickled_stat_cache, 'rb') as f:
            self.cache = pickle.load(f)

    def save(self, pickled_stat_cache):
        with open(pickled_stat_cache, 'wb') as f:
            pickle.dump(self.cache, f)
