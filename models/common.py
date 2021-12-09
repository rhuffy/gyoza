from dataclasses import dataclass
from typing import List
from enum import Enum


class ProgLang(Enum):
    RS = "rs"
    C = "c"
    PY = "py"


@dataclass
class Function:
    function_name: str
    function_language: ProgLang
    function_body: str


@dataclass
class Instance:
    instance_name: str
    instance_body: str


@dataclass
class FunctionOnInstance:
    function_data: Function
    instance_type_data: Instance


@dataclass
class Experience:
    function_on_instance: FunctionOnInstance
    stats: List[float]
