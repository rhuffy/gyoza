import argparse
import collections
import errno
import os
import random
import sys
from typing import List

from _typeshed import NoneType

from models.code_featurizers import (
    LSTMDocumentFeaturizer,
    LSTMNDocumentFeaturizer,
    LSTMStackFeaturizer,
)
from models.common import Experience, FunctionOnInstance
from models.embedding_models import LinearEmbedding
from models.gyoza_embedding import GyozaEmbedding
from models.instance_featurizers import DefaultInstanceFeaturizer
from models.model import GyozaModel
from program_analyzer import ProgramAnalyzer
from worker import WorkerInstance

LSTM = "lstm"
LSTMN = "lstmn"
NEURAL_STACK = "neural_stack"

INSTANCE_FEATURES = 10
PROGRAM_ANALYZER_FEATURES = 10
RUNTIME_STATISTICS = 8

ERROR_INVALID_NAME = 123


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def is_path_creatable(pathname: str) -> bool:
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def is_pathname_valid(pathname: str) -> bool:
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        _, pathname = os.path.splitdrive(pathname)

        root_dirname = os.environ.get('HOMEDRIVE', 'C:') if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)  # ...Murphy and her ironclad Law

        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)

            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError as exc:
        return False
    else:
        return True


def is_path_exists_or_creatable(pathname: str) -> bool:
    try:
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname)
        )
    except OSError:
        return False


parser = argparse.ArgumentParser()
parser.add_argument("--code-model", choices=[LSTM, LSTMN, NEURAL_STACK])
parser.add_argument("--instance-model", choices=["default"], default="default")
parser.add_argument("--test-functions", type=dir_path)
parser.add_argument("--experience-length", type=int)
parser.add_argument("--num-embeddings", type=int, default=10)
parser.add_argument("--embedding-dim", type=int, default=128)
parser.add_argument("--hidden-dim", type=int, default=512)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--model-path", type=is_path_exists_or_creatable)
parser.add_argument("--logging", type=bool, default=False)

args = parser.parse_args()

code_model_args = [args.num_embeddings, args.embedding_dim, args.hidden_dim, args.out_dim]
embedding_model_args = [
    args.out_dim + INSTANCE_FEATURES + PROGRAM_ANALYZER_FEATURES,
    RUNTIME_STATISTICS,
]


def create_model(code_model_args, embedding_model_args) -> GyozaModel:
    if args.code_model == LSTM:
        code_model = LSTMDocumentFeaturizer(*code_model_args)
    elif args.code_model == LSTMN:
        code_model = LSTMNDocumentFeaturizer(*code_model_args)
    else:
        code_model = LSTMStackFeaturizer(*code_model_args)

    program_analyzer = ProgramAnalyzer()
    instance_model = DefaultInstanceFeaturizer()
    embedding_model = LinearEmbedding(*embedding_model_args)

    embedding_model = GyozaEmbedding(code_model, instance_model, program_analyzer, embedding_model)
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
    for filename in os.listdir(os.path.dirname(__file__), "./instances"):
        if filename.endswith(".yml"):
            instances.append(filename[:-4])

    experience_buffer = collections.deque(maxlen=args.experience_length)

    # user definable
    def affinity(parameters: List[float]) -> float:
        return 0

    # not sure
    def stopping_condition(iters: int) -> bool:
        return False

    iter_count = 0
    for function in random_iter(functions):
        best_instance_idx, _ = max(
            [
                (i, model.predict(FunctionOnInstance(function, instance)))
                for i, instance in enumerate(instances)
            ],
            key=lambda x: affinity(x[1]),
        )
        best_instance = instances[best_instance_idx]
        res = worker.launch(function, best_instance)
        experience_buffer.append(Experience(function, best_instance, res))
        iter_count += 1
        if iter_count % args.experience_length == 0:
            # Thompson Sampling
            model = create_model(code_model_args, embedding_model_args)
            model.fit(
                random.choices(experience_buffer, args.experience_length),
                iter_count,
                args,
                logging=args.logging,
            )
        if stopping_condition(iter_count):
            break


if __name__ == "__main__":
    main()
