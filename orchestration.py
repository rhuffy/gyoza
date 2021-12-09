import argparse
import collections
import errno
import os
import random
import sys
from typing import List
import torch

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
from train import train_gyoza_thompson
from worker import WorkerInstance

LSTM = "lstm"
LSTMN = "lstmn"
NEURAL_STACK = "neural_stack"

INSTANCE_FEATURES = 4
PROGRAM_ANALYZER_FEATURES = 10
RUNTIME_STATISTICS = 9

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
parser.add_argument("--initial-random-experience-length", type=int)
parser.add_argument("--experience-length", type=int)
parser.add_argument("--embedding-dim", type=int, default=128)
parser.add_argument("--hidden-dim", type=int, default=512)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--model-path", type=str)
parser.add_argument("--stat-cache", type=str)
parser.add_argument("--logging", type=bool, default=False)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--N", type=int)
parser.add_argument("--K", type=int)
parser.add_argument("--max-iters", type=int)

args = parser.parse_args()

embedding_model_args = [
    args.out_dim + INSTANCE_FEATURES + PROGRAM_ANALYZER_FEATURES,
    RUNTIME_STATISTICS,
]


def create_model(code_model_args, embedding_model_args, program_analyzer, lang) -> GyozaModel:
    if args.code_model == LSTM:
        code_model = LSTMDocumentFeaturizer(*code_model_args)
    elif args.code_model == LSTMN:
        code_model = LSTMNDocumentFeaturizer(*code_model_args)
    else:
        code_model = LSTMStackFeaturizer(*code_model_args)

    instance_model = DefaultInstanceFeaturizer()
    embedding_model = LinearEmbedding(*embedding_model_args)

    embedding_model = GyozaEmbedding(
        code_model, instance_model, program_analyzer, embedding_model, lang
    )
    # (function, instance_info) -> compatability_stats
    return GyozaModel(embedding_model)


def get_all_functions(rel_path) -> List[str]:
    function_pointers = []
    for item in os.listdir(os.path.join(os.path.dirname(__file__), rel_path)):
        if item.endswith(".c"):
            function_pointers.append(item[:-2])
    function_pointers.append("mandelbrot")
    return function_pointers


def get_all_function_data(function_pointers: str) -> List[str]:
    results = []
    for function in function_pointers:
        if function == "mandelbrot":
            rel_path = f"./benchmarks/mandelbrot/src/main.rs"
        else:
            rel_path = f"./benchmarks/{function}.c"
        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            function_data = f.read()
        results.append(function_data)
    return results


def random_iter(items):
    items = list(items)
    while True:
        yield random.choice(items)


def custom_logger(log_str, verbose):
    if verbose:
        print(log_str)


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def main():
    custom_logger("Loading functions", args.verbose)
    functions = get_all_functions(args.test_functions)
    custom_logger("Loaded functions, Loading Program Analyzer", args.verbose)
    program_analyzer = ProgramAnalyzer()
    # if os.path.exists(args.stat_cache):
    #     program_analyzer.load(args.stat_cache)
    custom_logger("Loaded Program Analyzer", args.verbose)

    custom_logger("Building language...", args.verbose)
    lang = Lang()
    for function_data in get_all_function_data(functions):
        lang.add_sentence(function_data)

    custom_logger("Creating Model...", args.verbose)
    code_model_args = [lang.n_words, args.embedding_dim, args.hidden_dim, args.out_dim]

    custom_logger("Loading instance configs", args.verbose)
    instances = []
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "./instances")):
        if filename.endswith(".yml"):
            instances.append(filename[:-4])

    def affinity(parameters: List[float]) -> float:
        return parameters[0]

    with WorkerInstance() as worker:
        custom_logger("Beginning training", args.verbose)
        train_gyoza_thompson(
            worker,
            lambda: create_model(code_model_args, embedding_model_args, program_analyzer, lang),
            random_iter(functions),
            instances,
            affinity,
            args,
            lambda x: custom_logger(x, args.verbose),
            logging=args.logging,
        )
        # Save stat cache to disk
        program_analyzer.save(args.stat_cache)


if __name__ == "__main__":
    main()
