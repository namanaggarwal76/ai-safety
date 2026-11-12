"""injectbench: test whether AI agents resist instructions hidden in untrusted data."""
from .environment import Environment
from .goals import GOALS, Goal
from .models import AgentTask, AgentTrace, ToolCall, ToolSpec
from .scenario import AttackSurface, ForbiddenAction, InjectionPoint, Scenario
from .scenarios import SCENARIOS
from .techniques import BUILTIN as TECHNIQUES, Technique, load_techniques

__version__ = "0.1.0"
__all__ = [
    "AgentTask", "AgentTrace", "AttackSurface", "Environment", "ForbiddenAction", "GOALS", "Goal",
    "InjectionPoint", "SCENARIOS", "Scenario", "TECHNIQUES", "Technique", "ToolCall", "ToolSpec",
    "load_techniques",
]
