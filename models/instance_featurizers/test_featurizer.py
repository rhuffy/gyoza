import pytest

from .featurizer import DefaultInstanceFeaturizer


MOCK_YAML = """
cpu_requirement: 2.3
memory_requirement: 4.5
network_requirement: 8.7
"""


def test_instance_featurizer():
    featurizer = DefaultInstanceFeaturizer()
    result = featurizer(MOCK_YAML)
    assert result.tolist() == [2.3, 4.5, 8.7]
