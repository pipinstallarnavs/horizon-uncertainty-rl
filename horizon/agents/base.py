"""The agent interface every policy implements, scripted or LLM.

Keeping scripted baselines and the LLM behind one interface is what makes the
dry-run gate possible: the entire pipeline, metrics and figures run end to end on
free scripted agents before an API key is ever loaded (DESIGN.md T10.3).
"""
from dataclasses import dataclass, field

from horizon.env.market import CASH, HOLDINGS


@dataclass(frozen=True)
class Observation:
    """Everything the agent is told on a given day.

    Elapsed time is always visible by design: the study tests whether the agent
    *uses* the clock, not whether it can find it (README section 5.1).
    """

    day: int
    holding: int
    age: int
    wealth: float
    switches: tuple = ()
    spent: tuple = ()
    horizon: int = 100

    @property
    def holding_name(self) -> str:
        return HOLDINGS[self.holding]


@dataclass(frozen=True)
class Decision:
    """An agent's reply: the holding to be in today, plus optional probe belief."""

    action: int
    belief: float = None
    raw: str = ""
    error_class: str = "ok"
    repair_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thought_tokens: int = 0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def action_name(self) -> str:
        return HOLDINGS[self.action]


class Agent:
    """Base class. Subclasses implement `decide`."""

    name = "agent"
    uses_api = False

    def reset(self, episode: int) -> None:
        """Called once at the start of each episode."""

    def decide(self, obs: Observation) -> Decision:
        raise NotImplementedError


def initial_observation(horizon: int = 100) -> Observation:
    """Day 1: the agent is auto-entered holding cash (README section 3.6)."""
    return Observation(day=1, holding=CASH, age=0, wealth=1.0, horizon=horizon)
