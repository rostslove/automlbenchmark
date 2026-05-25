"""
AMLB adapter for LLM-based agentic AutoML frameworks.

Concrete framework definitions in resources/frameworks.yaml select the actual
agent through the _agent_framework parameter.
"""


def setup(*_args, **_kwargs):
    """No-op setup: external agent CLIs/repos are configured outside AMLB."""
    return None


def version():
    return "external"


def run(dataset, config):
    from .exec import run

    return run(dataset, config)
