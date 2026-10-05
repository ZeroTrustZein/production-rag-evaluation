"""Data models, enums, and schemas for agent workflow orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from production_rag_evaluation.models.schemas import (
    EvaluationMetrics,
    HallucinationFinding,
    RiskMetrics,
)


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class AgentRole(str, Enum):
    """Functional role designation of an agent in the swarm."""

    PLANNER = "planner"
    EVALUATOR = "evaluator"
    AUDITOR = "auditor"
    RISK_ANALYST = "risk_analyst"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


class AgentStatus(str, Enum):
    """Lifecycle state of an individual agent."""

    IDLE = "idle"
    BUSY = "busy"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


class WorkflowStatus(str, Enum):
    """State transition enum for overall workflow and individual steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepExecutionMode(str, Enum):
    """Execution dependency mode for workflow steps."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class AgentMessage(BaseModel):
    """Inter-agent communication message exchanged across the swarm."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    role: AgentRole
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class WorkflowStep(BaseModel):
    """Discrete step within a multi-agent evaluation workflow DAG."""

    model_config = ConfigDict(frozen=False)

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_role: AgentRole
    mode: StepExecutionMode = StepExecutionMode.SEQUENTIAL
    dependencies: List[str] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class WorkflowPlan(BaseModel):
    """Structured plan orchestrating multi-agent evaluation steps."""

    model_config = ConfigDict(frozen=False)

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    ticker: Optional[str] = None
    steps: List[WorkflowStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    estimated_duration_ms: float = 0.0

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Fetch step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_for_role(self, role: AgentRole) -> List[WorkflowStep]:
        """Fetch all steps assigned to a given agent role."""
        return [s for s in self.steps if s.agent_role == role]

    def get_independent_steps(self) -> List[WorkflowStep]:
        """Fetch all steps that have no dependencies and can execute immediately."""
        return [s for s in self.steps if not s.dependencies]

    def validate_dag(self) -> tuple[bool, Optional[str]]:
        """Validate that workflow steps form a valid directed acyclic graph."""
        step_ids = {s.step_id for s in self.steps}
        for s in self.steps:
            for dep in s.dependencies:
                if dep not in step_ids:
                    return False, f"Step '{s.name}' references unknown dependency '{dep}'"

        # Check for cycles using DFS
        adj: Dict[str, List[str]] = {s.step_id: list(s.dependencies) for s in self.steps}
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def has_cycle(u: str) -> bool:
            visited[u] = 1
            for v in adj.get(u, []):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and has_cycle(v):
                    return True
            visited[u] = 2
            return False

        for s_id in step_ids:
            if visited.get(s_id, 0) == 0:
                if has_cycle(s_id):
                    return False, "Cycle detected in workflow step dependencies"

        return True, None


class WorkflowExecutionResult(BaseModel):
    """Consolidated outcome of an executed multi-agent evaluation workflow."""

    model_config = ConfigDict(frozen=False)

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.COMPLETED
    plan: WorkflowPlan
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metrics: Optional[EvaluationMetrics] = None
    findings: List[HallucinationFinding] = Field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    total_execution_time_ms: float = 0.0
    error: Optional[str] = None


class AgentWorkflowConfig(BaseModel):
    """Runtime configuration for multi-agent evaluation workflows."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=10, ge=1, le=50)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    concurrency_limit: int = Field(default=5, ge=1, le=20)
    halt_on_high_severity_hallucination: bool = False
    auto_retry_failed_steps: bool = False
    max_retries: int = Field(default=2, ge=0, le=5)
