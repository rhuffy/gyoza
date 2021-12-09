from typing import List, NamedTuple
import os
import subprocess

from multiprocessing import Process, Event, Value

import docker

from pprint import pprint
import time
import yaml


IMAGE_NAME = "myimage"
TAG_NAME = "tag2"


POLLING_CADENCE = 0.01


class RunStatistics(NamedTuple):
    cpu_utilization: float
    memory_utilization: float
    network_rx: float
    network_tx: float

    @staticmethod
    def from_json(v) -> "RunStatistics":
        prev_cpu = v.get("precpu_stats").get("cpu_usage").get("total_usage")
        prev_system = v.get("precpu_stats").get("system_cpu_usage", 0.0)
        cpu_percent = 0.0
        cpu_delta = float(v.get("cpu_stats").get("cpu_usage").get("total_usage")) - float(prev_cpu)
        system_delta = float(v.get("cpu_stats").get("system_cpu_usage", 0)) - float(prev_system)

        if system_delta > 0.0 and cpu_delta > 0.0:
            cpu_percent = (cpu_delta / system_delta) * float(v["cpu_stats"]["online_cpus"]) * 100.0

        rx, tx = 0.0, 0.0

        for (_, net_stats) in v.get("networks").items():
            rx += net_stats.get("rx_bytes", 0.0)
            tx += net_stats.get("tx_bytes", 0.0)

        if int(v.get("memory_stats").get("limit")) != 0:
            mem_percent = (
                float(v["memory_stats"]["usage"]) / float(v["memory_stats"]["limit"]) * 100.0
            )
        else:
            mem_percent = 0.0

        return RunStatistics(
            cpu_utilization=cpu_percent,
            memory_utilization=mem_percent,
            network_rx=rx,
            network_tx=tx,
        )


# class CollectedRunStatistics(NamedTuple):
#     time_elapsed: float
#     max_cpu_utilization: float
#     average_cpu_utilization: float
#     max_memory_utilization: float
#     average_memory_utilization: float
#     max_network_rx: float
#     average_network_rx: float
#     max_network_tx: float
#     average_network_tx: float

#     @staticmethod
#     def from_run_stats(timestamp: float, stats: List[RunStatistics]):
#         max_cpu, sum_cpu = 0.0, 0.0
#         max_mem, sum_mem = 0.0, 0.0
#         max_rx, sum_rx = 0.0, 0.0
#         max_tx, sum_tx = 0.0, 0.0

#         for stat in stats:
#             sum_cpu += stat.cpu_utilization
#             max_cpu = max(max_cpu, stat.cpu_utilization)

#             sum_mem += stat.memory_utilization
#             max_mem = max(max_mem, stat.memory_utilization)

#             sum_rx += stat.network_rx
#             max_rx = max(max_rx, stat.network_rx)

#             sum_tx += stat.network_tx
#             max_tx = max(max_tx, stat.network_tx)

#             return [
#                 timestamp,
#                 max_cpu,
#                 sum_cpu / len(stats),
#                 max_mem,
#                 sum_mem / len(stats),
#                 max_rx,
#                 sum_rx / len(stats),
#                 max_tx,
#                 sum_tx / len(stats),
#             ]


def from_run_stats(timestamp: float, stats: List[RunStatistics]) -> List[float]:
    max_cpu, sum_cpu = 0.0, 0.0
    max_mem, sum_mem = 0.0, 0.0
    max_rx, sum_rx = 0.0, 0.0
    max_tx, sum_tx = 0.0, 0.0

    for stat in stats:
        sum_cpu += stat.cpu_utilization
        max_cpu = max(max_cpu, stat.cpu_utilization)

        sum_mem += stat.memory_utilization
        max_mem = max(max_mem, stat.memory_utilization)

        sum_rx += stat.network_rx
        max_rx = max(max_rx, stat.network_rx)

        sum_tx += stat.network_tx
        max_tx = max(max_tx, stat.network_tx)

        return [
            timestamp,
            max_cpu,
            sum_cpu / len(stats),
            max_mem,
            sum_mem / len(stats),
            max_rx,
            sum_rx / len(stats),
            max_tx,
            sum_tx / len(stats),
        ]


def launch_run(event, container_id, function_key, value):
    client = docker.from_env()
    start = time.time()
    client.containers.get(container_id).exec_run(f"./{function_key}")
    value.value = time.time() - start
    event.set()


class WorkerInstance:
    ## Launches jobs in a subprocess and collects some statistics on them
    def __init__(self) -> None:
        # collect names of available benchmarks
        self._benchmarks = set()
        for file_name in os.listdir(os.path.join(os.path.dirname(__file__), "./benchmarks")):
            if file_name.endswith(".c"):
                self._benchmarks.add(file_name[:-2])

        # manually add mandelbrot
        self._benchmarks.add("mandelbrot")

        self._client = docker.from_env()

    def launch(self, function_key: str, action_key: str) -> List[float]:
        if function_key not in self._benchmarks:
            raise Exception(f"Invalid benchmark {function_key}")

        event, value = Event(), Value("f", 0.0)
        with open(
            os.path.join(os.path.dirname(__file__), f"./instances/{action_key}.yml"), "r"
        ) as f:
            instance_config = yaml.safe_load(f)
            container = self._client.containers.run(
                f"{IMAGE_NAME}:{TAG_NAME}",
                detach=True,
                tty=True,
                mem_limit=instance_config.get("memory_size"),
                cpu_period=100000,
                cpu_quota=int(100000 * float(instance_config.get("num_cpus"))),
            )
        stats_generator = container.stats(decode=True, stream=True)

        launcher = Process(target=launch_run, args=[event, container.id, function_key, value])
        launcher.start()
        res = []

        while not event.wait(POLLING_CADENCE):
            res.append(next(stats_generator))

        res = [RunStatistics.from_json(collected_stat_json) for collected_stat_json in res]
        final_stats = from_run_stats(value.value, res)
        container.stop()
        return final_stats
