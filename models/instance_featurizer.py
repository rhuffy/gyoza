import yaml

from abc import ABC, abstractmethod
import numpy as np


class InstanceFeaturizer(ABC):
    @abstractmethod
    def featurize(self, node_yaml_str: str) -> np.ndarray:
        pass


class DefaultNodeFeaturizer(InstanceFeaturizer):
    def featurize(self, node_yaml_str: str) -> np.ndarray:
        try:
            node_yaml = yaml.safe_load(node_yaml_str)
            # TODO: actually generate docker_compose files appropriately here, etc.
            embeddings = [
                node_yaml.get("cpu_requirement", 0),
                node_yaml.get("memory_requirement", 0),
                node_yaml.get("network_requirement", 0),
            ]
            return np.array(embeddings)
        except yaml.YAMLError:
            print("Exception when parsing YAML e")
            return None
