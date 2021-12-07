import pytest
import numpy as np
import torch.nn as nn
import torch
from .model import GyozaModel, FunctionOnInstance
from .base_model import BaseModel


def test_model():
    mock_function_featurizer = lambda _: torch.tensor([1.0, 2.0])
    mock_instance_featurizer = lambda _: torch.tensor([3.0])
    embedding_model = nn.Linear(3, 1)

    base_model = BaseModel(mock_function_featurizer, mock_instance_featurizer, embedding_model)
    gyoza_model = GyozaModel(base_model)

    X_data = [
        FunctionOnInstance("abcdef", "gefjfkas"),
        FunctionOnInstance("afjkjasf", "akjfkdajkf"),
    ]
    y_data = [4.0, 7.0]

    gyoza_model.predict(X_data[0])
    gyoza_model.fit(X_data, y_data)
