import torch
import torch.nn as nn
from .types import FunctionOnInstance


class BaseModel(nn.Module):
    def __init__(self, function_featurizer, instance_featurizer, embedding_model):
        super().__init__()

        self._function_featurizer = function_featurizer
        self._instance_featurizer = instance_featurizer
        self._embedding_model = embedding_model

    def forward(self, computation_data: FunctionOnInstance):
        func = self._function_featurizer(computation_data.function_data)
        inst = self._instance_featurizer(computation_data.instance_type_data)

        embed = torch.cat([func, inst])
        return self._embedding_model(embed)
