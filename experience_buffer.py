from typing import List
from models.common import Experience
from collections import deque


class ExperienceBuffer:
    def __init__(self) -> None:
        self._buffer = deque()

    def add(self, experience: Experience):
        self._buffer.append(experience)

    def get_all(self) -> List[Experience]:
        return list(self._buffer)

    def clear_all(self):
        self._buffer = deque()
