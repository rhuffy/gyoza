from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import cuda

from .common import Experience, FunctionOnInstance
from .gyoza_embedding import GyozaEmbedding

# Goal here is (image, node) => predicted perf. score

# Treat action: image_features => (image_features, node_features, pref_features)

# experiences: ((image_features, node_features, pref_features), Performance

CUDA = cuda.is_available()


class GyozaModel:
    def __init__(self, embedding_model: GyozaEmbedding) -> None:
        super().__init__()
        self._embedding_model = embedding_model

    def fit(
        self,
        experience: List[Experience],
        samples_taken: int,
        args,
        learning_rate=1e-3,
        epochs=100,
        logging=False,
    ) -> float:
        # Below code is taken (w/ slight modification) from BAOForPostgreSQL Paper

        run = None
        if logging:
            run = wandb.init(project="6.887 Final Project", entity="milanb17")
            wandb.config = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "samples_taken": samples_taken,
                "args": args,
            }
            wandb.watch(self._embedding_model)

        optimizer = optim.Adam(self._embedding_model.parameters())
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            loss_accum = 0
            for i, e in enumerate(experience):
                y = torch.tensor([e.affinity_score])
                x = e.function_on_instance
                if CUDA:
                    y = y.cuda()
                y_pred = self._embedding_model(x)
                # print(y)
                # print(y_pred)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if logging and i % args.logging_interval == 0:
                    wandb.log({"loss": loss})

            loss_accum /= len(experience)
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
        torch.save(self._embedding_model.state_dict(), args.model_path)
        if logging:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(args.model_path)
            run.log_artifact(artifact)
            run.finish()

    def predict(self, computation_data: FunctionOnInstance) -> float:
        with torch.no_grad():
            self._embedding_model.eval()
            return self._embedding_model(computation_data).cpu().detach().numpy()
