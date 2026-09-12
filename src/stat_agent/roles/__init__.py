"""LLM roles: pure components with typed I/O and their own prompt."""

from stat_agent.roles.analyzer import Analysis, AnalyzerRole, BlockResult, ExecutionIssues
from stat_agent.roles.base import Role
from stat_agent.roles.generator import Generation, GeneratorRole, Segment, parse_segments
from stat_agent.roles.matcher import MatcherRole, MatchResult
from stat_agent.roles.method_card import MethodCardRole
from stat_agent.roles.planner import PlannerRole, PlanOutput, PlanResult, PlanStep
from stat_agent.roles.reflector import Reflection, ReflectorRole, retry_allowed, should_attempt_fix
from stat_agent.roles.summarizer import SummarizerRole, SummaryOutput
from stat_agent.roles.verifier import Verdict, VerifierOutput, VerifierRole

__all__ = [
    "Role",
    "PlannerRole",
    "PlanStep",
    "PlanOutput",
    "PlanResult",
    "MatcherRole",
    "MatchResult",
    "VerifierRole",
    "Verdict",
    "VerifierOutput",
    "GeneratorRole",
    "Generation",
    "Segment",
    "parse_segments",
    "ReflectorRole",
    "Reflection",
    "retry_allowed",
    "should_attempt_fix",
    "AnalyzerRole",
    "Analysis",
    "ExecutionIssues",
    "BlockResult",
    "MethodCardRole",
    "SummarizerRole",
    "SummaryOutput",
]
