import random
from typing import Callable, Iterator, List

from models.common import Experience, Function, FunctionOnInstance, Instance
from models.model import GyozaModel

from experience_buffer import ExperienceBuffer
from worker import WorkerInstance


def train_gyoza_thompson(
    worker: WorkerInstance,
    create_model: Callable[[], GyozaModel],
    functions: Iterator[Function],
    instances: List[Instance],
    affinity: Callable[[int, List[float]], float],
    args,
    logger,
    logging=False,
):
    N: int = args.N
    K: int = args.K
    max_iters: int = args.max_iters
    epochs = args.epochs

    experience_buffer = ExperienceBuffer()
    next_retrain = N
    for _ in range(next_retrain):
        x = next(functions)
        best_instance = random.choice(instances)
        logger(f"Running {x.function_name} on {best_instance.instance_name}")
        worker_stats = worker.launch(x, best_instance)
        experience_buffer.add(
            Experience(FunctionOnInstance(x, best_instance), affinity(best_instance, worker_stats))
        )

    model = create_model()
    logger("Fitting model 1")
    model.fit(experience_buffer.get_all(), 1, args, logging=logging, epochs=epochs)
    experience_buffer.clear_all()

    for i in range(2, max_iters + 2):
        for _ in range(next_retrain):
            x = next(functions)
            best_instance_idx, _ = max(
                [
                    (i, model.predict(FunctionOnInstance(x, instance)))
                    for i, instance in enumerate(instances)
                ],
                key=lambda x: x[1],
            )
            best_instance = instances[best_instance_idx]

            logger(f"Running {x.function_name} on {best_instance.instance_name}")
            worker_stats = worker.launch(x, best_instance)
            experience_buffer.add(
                Experience(
                    FunctionOnInstance(x, best_instance), affinity(best_instance, worker_stats)
                )
            )
        model = create_model()
        logger(f"Fitting model {i}")
        model.fit(experience_buffer.get_all(), i, args, logging=logging, epochs=epochs)
        experience_buffer.clear_all()
        next_retrain *= K
