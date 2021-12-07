from typing import Any, NamedTuple, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import cuda, tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Goal here is (image, node) => predicted perf. score

# Treat action: image_features => (image_features, node_features)

# experiences: ((image_features, node_features), Performance((image_features, node_features)))

CUDA = cuda.is_available()


class FunctionOnInstance(NamedTuple):
    function_data: str
    instance_type_data: str


def collate(zipped_list: List[Tuple[Any, Any]]):
    [first, second] = list(zip(*zipped_list))
    return first, tensor(second)


class GyozaModel:
    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    def fit(self, computation_data: List[FunctionOnInstance], performance_results: List[float]):
        # Below code is taken (w/ slight modification) from BAOForPostgreSQL Paper

        optimizer = optim.Adam(self._embedding_model.parameters())
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in zip(computation_data, performance_results):
                if CUDA:
                    y = y.cuda()
                y_pred = self._embedding_model(x)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(computation_data)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                print("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
        print("Stopped training after max epochs")

    def predict(self, computation_data: FunctionOnInstance) -> float:
        with torch.no_grad():
            self._embedding_model.eval()
            return self._embedding_model(computation_data).cpu().detach().numpy()
