"""Base abstract class and registry for swarm agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from production_rag_evaluation.agents.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    WorkflowStep,
)


class BaseAgent(ABC):
    """Abstract foundational agent defining communication and execution contracts."""

    def __init__(self, name: str, role: AgentRole) -> None:
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.inbox: List[AgentMessage] = []

    def receive_message(self, message: AgentMessage) -> None:
        """Receive and buffer an incoming inter-agent message."""
        self.inbox.append(message)

    def send_message(
        self,
        receiver: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Compose and return a message addressed to another agent."""
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            role=self.role,
            content=content,
            payload=payload or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state metadata."""
        return {
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "inbox_count": len(self.inbox),
        }

    @abstractmethod
    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned workflow step against the shared context."""


class AgentRegistry:
    """Thread-safe registry managing agent instances and role resolution."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent instance."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Retrieve agent by unique name."""
        return self._agents.get(name)

    def get_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """Retrieve all agents assigned to a specific role."""
        return [a for a in self._agents.values() if a.role == role]

    def list_agents(self) -> List[Dict[str, Any]]:
        """List registered agents with operational status."""
        return [a.to_dict() for a in self._agents.values()]

    def count(self) -> int:
        """Return count of registered agents."""
        return len(self._agents)

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()


_DEFAULT_REGISTRY: Optional[AgentRegistry] = None


def get_default_agent_registry() -> AgentRegistry:
    """Return singleton default agent registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = AgentRegistry()
    return _DEFAULT_REGISTRY
