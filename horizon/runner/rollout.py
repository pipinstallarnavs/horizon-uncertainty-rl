"""Drive one agent through one episode.

The exit time does not truncate the rollout here. Because returns are
deterministic and the exit only ends the episode without revealing anything, a
full-horizon holding sequence plus the exit pmf yields E[W_T] exactly -- so one
rollout scores against every exit realisation at once. The live LLM runner stops
at the realised exit day instead, because each day there costs an API call.
"""
from dataclasses import dataclass

from horizon.agents.base import Agent, Decision, Observation
from horizon.env.episode import wealth_path
from horizon.env.market import CASH, Market
from horizon.env.rules import burn, is_available


@dataclass
class EpisodeResult:
    holdings: list
    decisions: list
    wealth: list


def run_episode(agent: Agent, market: Market, horizon: int = 100, stop_at: int = None) -> EpisodeResult:
    """Roll the agent forward. `stop_at` truncates (used by the paid LLM runner)."""
    days = horizon if stop_at is None else min(stop_at, horizon)
    holdings, decisions = [], []
    holding, age, mask, wealth = CASH, 0, 0, 1.0
    switches = []

    for day in range(1, days + 1):
        obs = Observation(
            day=day,
            holding=holding,
            age=age,
            wealth=wealth,
            switches=tuple(switches),
            spent=tuple(t for t in range(market.n_assets) if not is_available(mask, t)),
            horizon=horizon,
        )
        decision = agent.decide(obs)
        action = decision.action

        if action != holding and not is_available(mask, action):
            # An agent may propose an illegal move; the environment refuses it
            # and the position is held instead. Logged so it can be counted.
            decision = Decision(
                action=holding,
                belief=decision.belief,
                raw=decision.raw,
                error_class="illegal_action",
                repair_used=decision.repair_used,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
                thought_tokens=decision.thought_tokens,
                latency_ms=decision.latency_ms,
            )
            action = holding

        if action == holding:
            age += 1
        else:
            mask = burn(mask, holding)
            switches.append((day, action))
            age = 1
        holding = action

        holdings.append(holding)
        decisions.append(decision)
        wealth = float(wealth_path(market, holdings)[-1])

    return EpisodeResult(holdings=holdings, decisions=decisions, wealth=list(wealth_path(market, holdings)))
