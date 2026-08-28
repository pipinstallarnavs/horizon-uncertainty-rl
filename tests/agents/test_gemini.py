"""Offline tests for the Gemini adapter. No network access, no API key."""
import pytest

from horizon.agents.base import Observation
from horizon.agents.gemini import (
    MIN_OUTPUT_TOKENS,
    AimdLimiter,
    GeminiAgent,
    GeminiClient,
    parse_reply,
)
from horizon.env.distributions import bimodal_exit
from horizon.env.market import CASH, default_market


class FakeClient:
    """Stands in for GeminiClient, replaying canned completions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, user, **kwargs):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        return {
            "text": reply.get("text", ""),
            "finish_reason": reply.get("finish_reason", "STOP"),
            "prompt_tokens": reply.get("prompt_tokens", 100),
            "completion_tokens": reply.get("completion_tokens", 10),
            "thought_tokens": reply.get("thought_tokens", 40),
            "thought_text": reply.get("thought_text", ""),
            "total_tokens": 150,
            "model_version": "gemini-3.7-flash",
            "latency_ms": 2100.0,
        }


def make_agent(replies, probe=True):
    return GeminiAgent(
        FakeClient(replies), default_market(), bimodal_exit(), condition="K", probe=probe
    )


def obs(day=27, holding=CASH, age=0):
    return Observation(day=day, holding=holding, age=age, wealth=1.0)


def test_parses_a_well_formed_two_line_reply():
    action, belief, error = parse_reply("BELIEF: 38\nACTION: SWITCH B", probe=True)
    assert (action, belief, error) == ("B", 38.0, "ok")


def test_parses_hold():
    assert parse_reply("BELIEF: 12.5\nACTION: HOLD", probe=True)[0] == "HOLD"


def test_parses_without_the_action_prefix():
    assert parse_reply("SWITCH CASH", probe=False)[0] == "CASH"


def test_rejects_an_unparseable_reply():
    action, _, error = parse_reply("I think I should probably wait a bit.", probe=True)
    assert action is None
    assert error == "format_error"


def test_belief_is_captured_even_when_the_action_is_malformed():
    _, belief, _ = parse_reply("BELIEF: 42\nACTION: maybe hold?", probe=True)
    assert belief == 42.0


def test_empty_reply_after_thinking_is_not_a_format_error():
    # The critical distinction: the model spent its whole budget thinking and
    # emitted nothing. That is a configuration bug and must never be counted
    # toward the episode-exclusion rule as if the model had failed to reason.
    agent = make_agent([{"text": "", "finish_reason": "MAX_TOKENS", "thought_tokens": 512}])
    decision = agent.decide(obs())
    assert decision.error_class == "empty_thinking_truncation"


def test_empty_reply_without_thinking_is_a_format_error():
    agent = make_agent([{"text": "", "finish_reason": "STOP", "thought_tokens": 0}])
    assert agent.decide(obs()).error_class == "format_error"


def test_a_malformed_reply_triggers_exactly_one_repair():
    agent = make_agent(
        [{"text": "hmm, let me think"}, {"text": "BELIEF: 30\nACTION: SWITCH A"}]
    )
    decision = agent.decide(obs())
    assert decision.error_class == "repaired"
    assert decision.repair_used
    assert decision.action == 0
    assert len(agent.client.calls) == 2


def test_two_failures_default_to_hold_and_flag_the_episode():
    agent = make_agent([{"text": "nope"}, {"text": "still nope"}])
    decision = agent.decide(obs(holding=1))
    assert decision.error_class == "format_error"
    assert decision.action == 1  # held


def test_repair_accumulates_token_counts_from_both_calls():
    agent = make_agent(
        [
            {"text": "bad", "prompt_tokens": 100, "completion_tokens": 5, "thought_tokens": 40},
            {"text": "ACTION: HOLD", "prompt_tokens": 120, "completion_tokens": 8,
             "thought_tokens": 60},
        ]
    )
    decision = agent.decide(obs())
    assert decision.prompt_tokens == 220
    assert decision.thought_tokens == 100


def test_thought_tokens_are_recorded_for_telemetry():
    agent = make_agent([{"text": "ACTION: HOLD", "thought_tokens": 173}])
    assert agent.decide(obs()).thought_tokens == 173


def test_client_refuses_an_output_budget_that_would_truncate_thinking():
    client = GeminiClient.__new__(GeminiClient)
    with pytest.raises(ValueError, match="thinking cannot be disabled"):
        GeminiClient.generate(client, "sys", "user", max_output_tokens=MIN_OUTPUT_TOKENS - 1)


def test_limiter_halves_on_throttle_and_creeps_back_up():
    limiter = AimdLimiter(start=16)
    limiter.on_throttle()
    assert limiter.limit == 8.0
    before = limiter.limit
    for _ in range(64):
        limiter.on_success()
    assert limiter.limit > before


def test_limiter_never_drops_below_one():
    limiter = AimdLimiter(start=2)
    for _ in range(20):
        limiter.on_throttle()
    assert limiter.limit >= 1.0


# --- reasoning traces ------------------------------------------------------ #
#
# includeThoughts=true was measured against the live endpoint: the deliberation
# comes back as extra parts flagged thought=true, ahead of the answer part.


def _payload(parts, thoughts=540):
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 12,
                          "thoughtsTokenCount": thoughts, "totalTokenCount": 612},
        "modelVersion": "gemini-3.7-flash",
    }


def test_thought_parts_never_leak_into_the_answer_text():
    # The trace habitually floats numbers it then rejects. Merged into the answer,
    # a belief the model considered and discarded would be parsed as its stated one.
    parsed = GeminiClient._parse(
        _payload([
            {"text": "Maybe BELIEF: 12? No -- day 12 of 56, so 44 remain.", "thought": True},
            {"text": "BELIEF: 44\nACTION: HOLD"},
        ]),
        latency_ms=1.0,
    )
    assert parsed["text"] == "BELIEF: 44\nACTION: HOLD"
    assert "Maybe BELIEF: 12" in parsed["thought_text"]
    assert parse_reply(parsed["text"], probe=True)[1] == 44.0


def test_multiple_thought_parts_are_joined_in_order():
    parsed = GeminiClient._parse(
        _payload([
            {"text": "first", "thought": True},
            {"text": "second", "thought": True},
            {"text": "ACTION: HOLD"},
        ]),
        latency_ms=1.0,
    )
    assert parsed["thought_text"] == "first\nsecond"


def test_a_reply_with_no_trace_still_parses():
    parsed = GeminiClient._parse(_payload([{"text": "ACTION: HOLD"}]), latency_ms=1.0)
    assert parsed["thought_text"] == ""
    assert parsed["text"] == "ACTION: HOLD"


def test_the_decision_carries_the_trace_through_to_the_log():
    agent = make_agent([
        {"text": "BELIEF: 38\nACTION: SWITCH B", "thought_text": "B is flat but safe."}
    ])
    assert agent.decide(obs()).thought_text == "B is flat but safe."


def test_a_repair_reports_the_trace_of_the_reply_that_was_kept():
    # The first attempt's reasoning explains a reply that was thrown away; the log
    # must show the thinking behind the answer that was actually used.
    agent = make_agent([
        {"text": "I think I will hold.", "thought_text": "rambling first attempt"},
        {"text": "ACTION: HOLD", "thought_text": "second attempt, correct format"},
    ])
    assert agent.decide(obs()).thought_text == "second attempt, correct format"


def test_thinking_level_and_budget_are_never_sent_together():
    # The endpoint rejects a request carrying both; level is the Gemini 3
    # parameter and wins when a cell specifies one.
    sent = {}

    class Capture(GeminiClient):
        def __init__(self):
            pass

        def generate(self, system, user, **kwargs):
            sent.update(kwargs)
            return {"text": "ACTION: HOLD", "thought_text": "", "finish_reason": "STOP",
                    "prompt_tokens": 1, "completion_tokens": 1, "thought_tokens": 1,
                    "total_tokens": 3, "model_version": "m", "latency_ms": 1.0}

    agent = GeminiAgent(Capture(), default_market(), bimodal_exit(),
                        thinking_level="high", thinking_budget=2048)
    agent.decide(obs())
    assert sent["thinking_level"] == "high"


def test_a_cell_defaults_to_no_thinking_level():
    from horizon.runner.orchestrator import build_cell

    assert build_cell()["thinking_level"] is None
