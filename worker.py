import signal
import sys
import time
from collections import defaultdict
from multiprocessing import Event, Process, Value
from threading import Thread
from typing import List, NamedTuple

import docker
import yaml

from models.common import Function, Instance, ProgLang

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


def launch_run(event, container_id, cmd, value):
    client = docker.from_env()
    start = time.time()
    client.containers.get(container_id).exec_run(cmd)
    value.value = time.time() - start
    event.set()


class WorkerInstance:
    ## Launches jobs in a subprocess and collects some statistics on them
    def __init__(self, image, tag) -> None:
        self._client = docker.from_env()

        self._container_cache = defaultdict(str)
        self._image = image
        self._tag = tag

    def launch(self, function: Function, instance: Instance) -> List[float]:
        event, value = Event(), Value("f", 0.0)
        cache_key = f"{function.function_name}-{instance.instance_name}"

        if cache_key not in self._container_cache:
            instance_config = yaml.safe_load(instance.instance_body)
            container = self._client.containers.run(
                f"{self._image}:{self._tag}",
                detach=True,
                tty=True,
                mem_limit=instance_config.get("memory_size"),
                cpu_period=100000,
                cpu_quota=int(100000 * float(instance_config.get("num_cpus"))),
            )
            self._container_cache[cache_key] = container.id
        else:
            container = self._client.containers.get(self._container_cache[cache_key])

        stats_generator = container.stats(decode=True, stream=True)

        if function.function_language == ProgLang.C:
            cmd = f"./c_benchmarks/bin/{function.function_name}"
        elif function.function_language == ProgLang.RS:
            cmd = f"./rust_benchmarks/bin/{function.function_name}"
        else:
            cmd = f"python3 ./python_benchmarks/{function.function_name}.py"

        launcher = Process(target=launch_run, args=[event, container.id, cmd, value])
        launcher.start()
        res = []

        while not event.wait(POLLING_CADENCE):
            res.append(next(stats_generator))

        res = [RunStatistics.from_json(collected_stat_json) for collected_stat_json in res]
        final_stats = from_run_stats(value.value, res)
        return final_stats

    def _handle_interrupt(self, signal_received, frame):
        sys.exit()

    def __enter__(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        return self

    def __exit__(self, *args):
        def close_container(containers, container_id):
            container = containers.get(container_id)
            container.stop()
            container.remove()

        threads = []
        for container_id in self._container_cache.values():
            thread = Thread(target=close_container, args=(self._client.containers, container_id))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
