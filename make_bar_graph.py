import os
import csv
import json

from utils import file_relative_path
from .worker import WorkerInstance

with WorkerInstance("myimage", "tag2") as inst:
    header = ["benchmark", "instance_name", "time", "json_data"]

    function_types = ["simple_bench", "bench_2"]
    instance_types = [f"instance{i}" for i in range(1, 7)]

    with open(file_relative_path(__file__, "./data.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        data = []
        for func_type in function_types:
            for instance_type in instance_types:
                print(f"Launching run for function {func_type} on instance {instance_type}...")
                result = inst.launch(func_type, instance_type)
                data.append([func_type, instance_types, result[0], json.dumps(result)])
        writer.writerows(data)
