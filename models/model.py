from typing import Any, NamedTuple, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
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
    def __init__(
        self, function_featurizer, instance_featurizer, embedding_model, performance_metric
    ) -> None:
        super().__init__()
        self._function_featurizer = function_featurizer
        self._instance_featurizer = instance_featurizer
        self._embedding_model = embedding_model
        self._performance_metric = performance_metric

    def fit(self, computation_data: List[FunctionOnInstance], performance_results: List[float]):
        function_on_instance_embeddings = [
            np.concatenate(
                self._function_featurizer(d.function_data),
                self._instance_featurizer(d.instance_type_data),
            )
            for d in computation_data
        ]

        # Below code is taken (w/ slight modification) from BAOForPostgreSQL Paper

        data_pairs = list(zip(function_on_instance_embeddings, performance_results))
        dataset = DataLoader(data_pairs, batch_size=16, shuffle=True, collate_fn=collate)

        optimizer = optim.Adam(self._embedding_model.parameters())
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda()
                y_pred = self._embedding_model(x)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
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

    def predict(self, computation_data: List[FunctionOnInstance]) -> float:
        function_on_instance_embeddings = [
            np.concatenate(
                self._function_featurizer(d.function_data),
                self._instance_featurizer(d.instance_type_data),
            )
            for d in computation_data
        ]

        self._embedding_model.eval()
        return self._embedding_model(function_on_instance_embeddings).cpu().detach().numpy()
