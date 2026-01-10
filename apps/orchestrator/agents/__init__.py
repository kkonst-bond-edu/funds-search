"""Agent fleet infrastructure for specialized AI agents."""

from apps.orchestrator.agents.base import BaseAgent
from apps.orchestrator.agents.matchmaker import MatchmakerAgent
from apps.orchestrator.agents.vacancy_analyst import VacancyAnalyst

__all__ = ["BaseAgent", "MatchmakerAgent", "VacancyAnalyst"]
