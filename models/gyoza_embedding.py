import torch
import torch.nn as nn
from .common import FunctionOnInstance
import os


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


class GyozaEmbedding(nn.Module):
    def __init__(
        self, function_featurizer, instance_featurizer, program_analyzer, embedding_model, lang
    ):
        super().__init__()

        self._function_featurizer = function_featurizer
        self._instance_featurizer = instance_featurizer
        self._program_analyzer = program_analyzer
        self._embedding_model = embedding_model
        self._lang = lang

    def forward(self, computation_data: FunctionOnInstance):
        if computation_data.function_data == "mandelbrot":
            rel_path = f"../benchmarks/mandelbrot/src/main.rs"
        else:
            rel_path = f"../benchmarks/{computation_data.function_data}.c"
        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            function_data = f.read()

        self._lang.add_sentence(function_data)
        input_tensor = tensor_from_sentence(self._lang, function_data)

        func = self._function_featurizer(input_tensor)
        anal = torch.tensor(self._program_analyzer())
        inst = torch.tensor(self._instance_featurizer(
            computation_data.instance_type_data))

        embed = torch.cat([func, anal, inst]).float()
        return self._embedding_model(embed)
