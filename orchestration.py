import argparse
import os
import random
import yaml
from typing import List, Tuple

from models.code_featurizers import (
    LSTMDocumentFeaturizer,
    LSTMNDocumentFeaturizer,
    LSTMStackFeaturizer,
)
from models.common import Function, Instance, ProgLang
from models.embedding_models import LinearEmbedding
from models.gyoza_embedding import GyozaEmbedding
from models.instance_featurizers import DefaultInstanceFeaturizer
from models.model import GyozaModel
from program_analyzer import ProgramAnalyzer
from train import train_gyoza_thompson
from utils import file_relative_path
from worker import WorkerInstance

LSTM = "lstm"
LSTMN = "lstmn"
NEURAL_STACK = "neural_stack"

INSTANCE_FEATURES = 4
PROGRAM_ANALYZER_FEATURES = 10
RUNTIME_STATISTICS = 10

ERROR_INVALID_NAME = 123


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser()
parser.add_argument("--code-model", choices=[LSTM, LSTMN, NEURAL_STACK])
parser.add_argument("--instance-model", choices=["default"], default="default")
parser.add_argument("--test-functions", type=dir_path)
parser.add_argument("--initial-random-experience-length", type=int)
parser.add_argument("--experience-length", type=int)
parser.add_argument("--embedding-dim", type=int, default=128)
parser.add_argument("--hidden-dim", type=int, default=512)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--embedding-hidden", type=int, default=10)
parser.add_argument("--model-path", type=str)
parser.add_argument("--stat-cache", type=str)
parser.add_argument("--logging", type=bool, default=False)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--N", type=int)
parser.add_argument("--K", type=int)
parser.add_argument("--max-iters", type=int)
parser.add_argument("--logging-interval", type=int, default=1)
parser.add_argument("--docker-image", type=str, default="myimage")
parser.add_argument("--docker-tag", type=str, default="tag2")

args = parser.parse_args()

embedding_model_args = [
    args.out_dim + INSTANCE_FEATURES + PROGRAM_ANALYZER_FEATURES,
    args.embedding_hidden,
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


def get_all_functions_and_data(rel_path) -> Tuple[List[Function], List[str]]:
    function_pointers = []
    benchmark_dir_path = file_relative_path(__file__, rel_path)
    for dir_name in os.listdir(benchmark_dir_path):
        dir_path = os.path.join(benchmark_dir_path, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                source_code_path = os.path.join(benchmark_dir_path, dir_name, file_name)
                if dir_name == "c_benchmarks" and file_name.endswith(".c"):
                    with open(source_code_path, "r") as f:
                        function_body = f.read()
                    function_pointers.append(Function(file_name[:-2], ProgLang.C, function_body))
                elif dir_name == "python_benchmarks" and file_name.endswith(".py"):
                    with open(source_code_path, "r") as f:
                        function_body = f.read()
                    function_pointers.append(Function(file_name[:-3], ProgLang.PY, function_body))
                elif dir_name == "rust_benchmarks" and os.path.isdir(source_code_path):
                    source_code_path = os.path.join(source_code_path, "./src/main.rs")
                    with open(source_code_path, "r") as f:
                        function_body = f.read()
                    function_pointers.append(Function(file_name, ProgLang.RS, function_body))
    return function_pointers


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
    functions = get_all_functions_and_data(args.test_functions)
    custom_logger("Loaded functions, Loading Program Analyzer", args.verbose)
    program_analyzer = ProgramAnalyzer()
    # if os.path.exists(args.stat_cache):
    #     program_analyzer.load(args.stat_cache)
    custom_logger("Loaded Program Analyzer", args.verbose)

    custom_logger("Building language...", args.verbose)
    lang = Lang()
    for f in functions:
        lang.add_sentence(f.function_body)

    custom_logger("Creating Model...", args.verbose)
    code_model_args = [lang.n_words, args.embedding_dim, args.hidden_dim, args.out_dim]

    custom_logger("Loading instance configs", args.verbose)
    instances, instances_path = [], file_relative_path(__file__, "./instances")
    for filename in os.listdir(instances_path):
        if filename.endswith(".yml"):
            with open(os.path.join(instances_path, filename), "r") as f:
                instances.append(Instance(filename[:-4], f.read()))

    def affinity(instance: Instance, parameters: List[float]) -> float:
        cost = float(yaml.safe_load(instance.instance_body).get("cost"))
        return -parameters[1] - 0.1 * cost if parameters[0] < 1.0 else float('-inf')

    with WorkerInstance(args.docker_image, args.docker_tag) as worker:
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
