"""
AMLB adapter for LLM-based agentic AutoML frameworks.

Concrete framework definitions in resources/frameworks.yaml select the actual
agent through the _agent_framework parameter.
"""


def run(dataset, config):
    from .exec import run

    return run(dataset, config)
