"""Vertex AI adapter for gemini-3.7-flash.

Everything here is shaped by three properties measured against the live endpoint
on 2026-08-27:

1. The model is available at location=global only; us-central1 returns 404.
2. Thinking cannot be disabled. thinkingBudget has no useful effect here, and
   the reason -- established later -- is that it is the Gemini 2.5 parameter;
   Gemini 3 takes thinkingLevel instead. A tight maxOutputTokens therefore
   yields a 200 OK carrying finishReason=MAX_TOKENS, ZERO text parts and a
   nonzero thoughtsTokenCount -- a silent failure that a naive parser books as a
   format error, which then counts toward the 5% episode-exclusion rule. It gets
   its own error class so an infrastructure misconfiguration can never be
   misreported as the model failing at temporal reasoning.
3. Throughput ceilings vary by client and network, so the rate limiter discovers
   the ceiling by AIMD rather than hardcoding a number measured elsewhere.
4. thinkingConfig.includeThoughts=true DOES work, and returns the deliberation
   as extra parts flagged thought=true alongside the answer part. Since the
   thoughts are billed either way, refusing to read them would throw away the
   only direct evidence of *why* a trade was made. They are captured into their
   own field and never concatenated into the answer text: the grammar parser
   must never see a BELIEF the model merely considered mid-thought.

   What comes back is a thought *summary*, not the raw chain of thought. No
   current Gemini model returns raw reasoning: five were tested against this
   endpoint and all returned summaries, with Gemini 3 adding an encrypted
   thoughtSignature. The one model that did expose raw thinking,
   gemini-2.0-flash-thinking-exp-01-21, is retired and 404s. This is a ceiling
   on the evidence, not a setting anyone can change.

5. thinkingLevel ("low" | "high"; "minimal" is rejected by this model) does
   control reasoning depth, measured over four samples of one fixed prompt:

       low      median  232 thought tokens, 1,620 trace characters
       default  median  435 thought tokens, 2,438 trace characters
       high     median 1272 thought tokens, 2,432 trace characters

   Note the last column. Raising the level buys roughly three times the
   deliberation but not a longer trace to read, because the summary length is
   near-fixed. It is a treatment knob, not an observability one, so it belongs
   in the cell metadata and cannot be changed mid-cell.
"""
import random
import re
import threading
import time

import requests
from google.oauth2 import service_account
from requests.adapters import HTTPAdapter

import google.auth.transport.requests as google_requests
from horizon.agents.base import Agent, Decision, Observation
from horizon.agents.prompts import daily_prompt, system_prompt
from horizon.env.market import HOLDINGS

MODEL_ID = "gemini-3.7-flash"
LOCATION = "global"
ENDPOINT_HOST = "aiplatform.googleapis.com"
SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

# Never lower this. See module docstring point 2.
MIN_OUTPUT_TOKENS = 512

# What we actually ask for. Measured on the first cells: at 512 the thought-token
# histogram piled up against the ceiling (more decisions in the top bucket than
# the one below it) and traces ended mid-sentence, so the budget -- not the
# model -- was deciding when reasoning stopped. Thought tokens bill as output,
# but a truncated deliberation is unusable evidence, and the run is not
# budget-constrained.
DEFAULT_OUTPUT_TOKENS = 4096

# Ask for the reasoning trace. This does not change what the model answers --
# summaries are synthesised from thoughts the model already spent -- and the
# tokens bill identically, so it is pure observability.
INCLUDE_THOUGHTS = True

ACTION_RE = re.compile(r"^\s*(?:ACTION:\s*)?(HOLD|SWITCH\s+(A|B|C|CASH))\s*$", re.IGNORECASE)
BELIEF_RE = re.compile(r"BELIEF:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


class TokenSource:
    """Mints and refreshes Vertex access tokens from a service-account key.

    ADC is deliberately not used: it was broken on the development machine and
    the fleet authenticates as a service account anyway.
    """

    def __init__(self, key_path: str, refresh_margin: float = 300.0):
        self._credentials = service_account.Credentials.from_service_account_file(
            key_path, scopes=list(SCOPES)
        )
        self._margin = refresh_margin
        self._lock = threading.Lock()

    def token(self) -> str:
        with self._lock:
            expiry = self._credentials.expiry
            stale = expiry is None or (expiry.timestamp() - time.time()) < self._margin
            if stale or not self._credentials.token:
                self._credentials.refresh(google_requests.Request())
            return self._credentials.token


class AimdLimiter:
    """Additive-increase / multiplicative-decrease concurrency control.

    The measured ceiling (~9 req/s from a laptop) will differ from a GCE VM, so
    the limiter finds the real ceiling and tracks it as it drifts instead of
    trusting a constant measured somewhere else.
    """

    def __init__(self, start: int = 8, minimum: int = 1, maximum: int = 256):
        self.limit = float(start)
        self.minimum = minimum
        self.maximum = maximum
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(start)
        self._issued = start

    def on_success(self) -> None:
        with self._lock:
            if self.limit < self.maximum:
                self.limit = min(self.maximum, self.limit + 1.0 / max(self.limit, 1.0))
                self._resize()

    def on_throttle(self) -> None:
        with self._lock:
            self.limit = max(self.minimum, self.limit / 2.0)
            self._resize()

    def _resize(self) -> None:
        target = max(self.minimum, int(self.limit))
        while self._issued < target:
            self._semaphore.release()
            self._issued += 1

    def __enter__(self):
        self._semaphore.acquire()
        return self

    def __exit__(self, *exc):
        self._semaphore.release()


class GeminiError(Exception):
    pass


class GeminiClient:
    """Thin transport layer: one request, retries, token accounting."""

    def __init__(
        self,
        project: str,
        token_source: TokenSource,
        limiter: AimdLimiter = None,
        max_retries: int = 5,
        timeout: float = 180.0,
        pool: int = 400,
    ):
        self.project = project
        self.tokens = token_source
        self.limiter = limiter or AimdLimiter()
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.mount(
            "https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
        )
        self.url = (
            f"https://{ENDPOINT_HOST}/v1/projects/{project}/locations/{LOCATION}"
            f"/publishers/google/models/{MODEL_ID}:generateContent"
        )

    def generate(self, system: str, user: str, temperature: float = 0.0,
                 max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
                 thinking_budget: int = None, thinking_level: str = None) -> dict:
        """One completion. Returns a dict with text, token counts and latency."""
        if max_output_tokens < MIN_OUTPUT_TOKENS:
            raise ValueError(
                f"max_output_tokens must be >= {MIN_OUTPUT_TOKENS}: thinking cannot be "
                "disabled on this model and a tighter budget returns empty text"
            )
        config = {"maxOutputTokens": max_output_tokens, "temperature": temperature}
        thinking = {}
        # Mutually exclusive: the endpoint rejects a request carrying both.
        if thinking_level is not None:
            thinking["thinkingLevel"] = thinking_level
        elif thinking_budget is not None:
            thinking["thinkingBudget"] = thinking_budget
        if INCLUDE_THOUGHTS:
            thinking["includeThoughts"] = True
        if thinking:
            config["thinkingConfig"] = thinking
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }

        last_error = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                with self.limiter:
                    response = self.session.post(
                        self.url,
                        headers={
                            "Authorization": f"Bearer {self.tokens.token()}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=self.timeout,
                    )
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            latency_ms = (time.time() - started) * 1000.0

            if response.status_code == 429:
                self.limiter.on_throttle()
                last_error = GeminiError("429 rate limited")
                self._backoff(attempt)
                continue
            if response.status_code >= 500:
                last_error = GeminiError(f"{response.status_code} {response.text[:200]}")
                self._backoff(attempt)
                continue
            if response.status_code != 200:
                raise GeminiError(f"{response.status_code} {response.text[:400]}")

            self.limiter.on_success()
            return self._parse(response.json(), latency_ms)

        raise GeminiError(f"exhausted {self.max_retries} retries: {last_error}")

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(30.0, (2**attempt) * 0.5) * (0.5 + random.random()))

    @staticmethod
    def _parse(payload: dict, latency_ms: float) -> dict:
        candidates = payload.get("candidates") or [{}]
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        # A part carrying thought=true is deliberation, not the reply. Merging
        # the two would let a belief the model floated and then rejected be
        # parsed as its stated belief.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thought_text = "\n".join(
            p.get("text", "") for p in parts if p.get("thought") and p.get("text")
        )
        usage = payload.get("usageMetadata", {})
        return {
            "text": text,
            "thought_text": thought_text,
            "finish_reason": candidate.get("finishReason"),
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "thought_tokens": usage.get("thoughtsTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
            "model_version": payload.get("modelVersion"),
            "latency_ms": latency_ms,
        }


def parse_reply(text: str, probe: bool):
    """Parse a completion into (action_index, belief, error_class).

    Returns action=None when the grammar does not match, so the caller can run
    the single repair re-ask specified in DESIGN.md P2.
    """
    belief = None
    if probe:
        match = BELIEF_RE.search(text)
        if match:
            belief = float(match.group(1))

    for line in reversed([ln for ln in text.strip().splitlines() if ln.strip()]):
        match = ACTION_RE.match(line.strip())
        if not match:
            continue
        verb = match.group(1).upper()
        if verb == "HOLD":
            return "HOLD", belief, "ok"
        target = match.group(2).upper()
        return target, belief, "ok"
    return None, belief, "format_error"


class GeminiAgent(Agent):
    """The agent under test. One API call per day, with one repair re-ask."""

    name = MODEL_ID
    uses_api = True

    def __init__(
        self,
        client: GeminiClient,
        market,
        dist,
        condition: str = "K",
        variant: str = "table",
        probe: bool = True,
        history: str = "compressed",
        temperature: float = 0.0,
        thinking_budget: int = None,
        thinking_level: str = None,
        true_exit_day: int = None,
        max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    ):
        self.client = client
        self.market = market
        self.probe = probe
        self.history = history
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.system = system_prompt(
            market, dist, condition, variant, probe, true_exit_day
        )

    def decide(self, obs: Observation) -> Decision:
        user = daily_prompt(obs, probe=self.probe, history=self.history)
        result = self.client.generate(
            self.system,
            user,
            temperature=self.temperature,
            thinking_budget=self.thinking_budget,
            thinking_level=self.thinking_level,
            max_output_tokens=self.max_output_tokens,
        )

        # See module docstring point 2: an empty reply that spent thought tokens
        # is a configuration bug, never a reasoning failure.
        if not result["text"].strip():
            error = (
                "empty_thinking_truncation"
                if result["finish_reason"] == "MAX_TOKENS" and result["thought_tokens"]
                else "format_error"
            )
            return self._decision(obs.holding, None, result, error, repaired=False)

        action, belief, error = parse_reply(result["text"], self.probe)
        repaired = False

        if action is None:
            repaired = True
            repair = (
                f"{user}\n\nYour previous reply did not match the required format:\n"
                f"{result['text'][:200]}\n\nReply again using EXACTLY the required format."
            )
            retry = self.client.generate(
                self.system,
                repair,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                thinking_level=self.thinking_level,
                max_output_tokens=self.max_output_tokens,
            )
            for key in ("prompt_tokens", "completion_tokens", "thought_tokens"):
                result[key] += retry[key]
            result["text"] = retry["text"]
            result["thought_text"] = retry.get("thought_text", "")
            result["latency_ms"] += retry["latency_ms"]
            action, belief, error = parse_reply(retry["text"], self.probe)

        if action is None:
            # Second failure: default to HOLD and flag the episode (P2).
            return self._decision(obs.holding, belief, result, "format_error", repaired)

        target = obs.holding if action == "HOLD" else HOLDINGS.index(action)
        return self._decision(target, belief, result, "repaired" if repaired else "ok", repaired)

    @staticmethod
    def _decision(action, belief, result, error_class, repaired) -> Decision:
        return Decision(
            action=action,
            belief=belief,
            raw=result["text"],
            thought_text=result.get("thought_text", ""),
            error_class=error_class,
            repair_used=repaired,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            thought_tokens=result["thought_tokens"],
            latency_ms=result["latency_ms"],
            extra={"model_version": result.get("model_version")},
        )
