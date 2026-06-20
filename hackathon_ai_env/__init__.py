"""Hackathon-ready multi-agent environment with Q-learning control."""

from .environment import HackathonAIEnvironment
from .scenarios import QueryScenario, default_scenarios

__all__ = ["HackathonAIEnvironment", "QueryScenario", "default_scenarios"]
