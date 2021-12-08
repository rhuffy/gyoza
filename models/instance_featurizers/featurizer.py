import yaml

import os
from abc import ABC, abstractmethod
import numpy as np


class InstanceFeaturizer(ABC):
    @abstractmethod
    def __call__(self, instance_name: str) -> np.ndarray:
        pass


class DefaultInstanceFeaturizer(InstanceFeaturizer):
    def _mem_size_to_float(self, memory_str: str) -> float:
        prefix, suffix = float(memory_str[0]), memory_str[1]
        if suffix == "m":
            return prefix * 1.0e6
        elif suffix == "g":
            return prefix * 1.0e9
        else:
            raise Exception(f"Unsupported memory_str {memory_str}")

    def __call__(self, instance_name: str) -> np.ndarray:
        try:
            with open(
                os.path.join(os.path.dirname(__file__), f"../../instances/{instance_name}.yml"), "r"
            ) as f:
                node_yaml = yaml.safe_load(f)
                embeddings = [
                    float(node_yaml.get("num_cpus", 1)),
                    self._mem_size_to_float(node_yaml.get("memory_size", "1g")),
                    float(node_yaml.get("network_ops", 0)),
                    float(node_yaml.get("cost", 100)),
                ]
                return np.array(embeddings)
        except yaml.YAMLError:
            print(f"Exception when parsing YAML \n{instance_name}")
            return None
