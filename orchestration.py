import argparse
import collections
import os
import random
from typing import List

from _typeshed import NoneType

from models.code_featurizers import (LSTMDocumentFeaturizer,
                                     LSTMNDocumentFeaturizer,
                                     LSTMStackFeaturizer)
from models.common import Experience, FunctionOnInstance
from models.gyoza_embedding import GyozaEmbedding
from models.instance_featurizers import DefaultInstanceFeaturizer
from models.model import GyozaModel
from models.embedding_models import LinearEmbedding
from worker import WorkerInstance

LSTM = "lstm"
LSTMN = "lstmn"
NEURAL_STACK = "neural_stack"


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser()
parser.add_argument("--code-model", choices=[LSTM, LSTMN, NEURAL_STACK])
parser.add_argument("--instance-model", choices=["default"], default="default")
parser.add_argument("--test-functions", type=dir_path)
parser.add_argument("--experience-length", type=int)
parser.add_argument("--num-embeddings", type=int, default=10)
parser.add_argument("--embedding-dim", type=int, default=128)
parser.add_argument("--hidden-dim", type=int, default=512)
parser.add_argument("--out-dim", type=int, default=32)

args = parser.parse_args()
code_model_args = [args.num_embeddings,
                   args.embedding_dim, args.hidden_dim, args.out_dim]
embedding_model_args = []


def create_model(code_model_args, embedding_model_args):
    if args.code_model == LSTM:
        code_model = LSTMDocumentFeaturizer(*code_model_args)
    elif args.code_model == LSTMN:
        code_model = LSTMNDocumentFeaturizer(*code_model_args)
    else:
        code_model = LSTMStackFeaturizer(*code_model_args)

    instance_model = DefaultInstanceFeaturizer()
    embedding_model = LinearEmbedding(*embedding_model_args)

    embedding_model = GyozaEmbedding(code_model, instance_model, None)
    # (function, instance_info) -> compatability_stats
    return GyozaModel(embedding_model)


def get_all_functions(_dir: str) -> List[NoneType]:
    return []


def random_iter(items):
    items = list(items)
    while True:
        yield random.choice(items)


def main():
    functions = get_all_functions(args.test_functions)
    model = create_model(code_model_args, embedding_model_args)
    worker = WorkerInstance()
    instances = []

    experience_buffer = collections.deque(maxlen=args.experience_length)

    # user definable
    def affinity(parameters: List[float]) -> float:
        return 0

    # not sure
    def stopping_condition(iters: int) -> bool:
        return False

    iter_count = 0
    for function in random_iter(functions):
        best_instance = max([model.predict(FunctionOnInstance(
            function, instance)) for instance in instances], key=affinity)
        res = worker.launch(function, best_instance)
        experience_buffer.append(Experience(function, best_instance, res))
        iter_count += 1
        if iter_count % args.experience_length == 0:
            # Thompson Sampling
            model = create_model(code_model_args)
            model.fit(random.choices(
                experience_buffer, args.experience_length))
        if stopping_condition(iter_count):
            break


if __name__ == "__main__":
    main()
