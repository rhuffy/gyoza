from dataclasses import dataclass
from typing import List


@dataclass
class FunctionOnInstance:
    function_data: str
    instance_type_data: str


@dataclass
class Experience:
    function: str
    instance: str
    stats: List[float]
