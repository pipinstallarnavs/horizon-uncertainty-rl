"""Database access for the run.

The leasing query is the whole concurrency design. One atomic statement gives
crash resume, idempotent restart, arbitrary worker fan-out and no double-spend,
which is what lets the researcher close the laptop mid-run.
"""
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_LEASE_MINUTES = 20

# Claim the next pending unit, or steal one whose lease has expired because its
# worker died. SKIP LOCKED is what guarantees two workers never take the same
# episode without serialising the whole queue.
CLAIM_SQL = """
UPDATE work_units SET
    status = 'leased',
    lease_owner = %(owner)s,
    lease_expires_at = %(expires)s,
    attempts = attempts + 1
WHERE (run_id, episode) = (
    SELECT run_id, episode FROM work_units
    WHERE run_id = %(run_id)s
      AND (status = 'pending'
           OR (status = 'leased' AND lease_expires_at < now()))
    ORDER BY episode
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING episode, attempts;
"""


class Repository:
    """Thin wrapper over psycopg. Import of psycopg is deferred so the whole
    scripted-agent dry run works without a database installed."""

    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.environ.get("HORIZON_DSN", "postgresql:///horizon")
        self._connection = None

    def connect(self):
        if self._connection is None or self._connection.closed:
            import psycopg

            self._connection = psycopg.connect(self.dsn, autocommit=True)
        return self._connection

    @contextmanager
    def cursor(self):
        with self.connect().cursor() as cur:
            yield cur

    def init_schema(self) -> None:
        with self.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    # ---- runs -------------------------------------------------------------

    def create_run(self, run_id: str, cell: dict, horizon: int, v_star: float,
                   model_snapshot: str = None, spend_cap_usd: float = None) -> None:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO runs (run_id, cell, horizon, v_star, model_snapshot, spend_cap_usd)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (run_id) DO NOTHING""",
                (run_id, json.dumps(cell), horizon, v_star, model_snapshot, spend_cap_usd),
            )

    def seed_work_units(self, run_id: str, episodes: int) -> None:
        """Idempotent: re-seeding an existing run adds nothing and loses nothing."""
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO work_units (run_id, episode) VALUES (%s, %s)
                   ON CONFLICT (run_id, episode) DO NOTHING""",
                [(run_id, i) for i in range(episodes)],
            )

    def set_run_status(self, run_id: str, status: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                """UPDATE runs SET status = %s,
                   finished_at = CASE WHEN %s IN ('complete', 'aborted') THEN now() ELSE NULL END
                   WHERE run_id = %s""",
                (status, status, run_id),
            )

    # ---- leasing ----------------------------------------------------------

    def claim(self, run_id: str, owner: str, lease_minutes: int = DEFAULT_LEASE_MINUTES):
        expires = datetime.now(timezone.utc) + timedelta(minutes=lease_minutes)
        with self.cursor() as cur:
            cur.execute(CLAIM_SQL, {"run_id": run_id, "owner": owner, "expires": expires})
            row = cur.fetchone()
        return None if row is None else {"episode": row[0], "attempts": row[1]}

    def complete(self, run_id: str, episode: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                """UPDATE work_units SET status = 'done', lease_owner = NULL,
                   lease_expires_at = NULL WHERE run_id = %s AND episode = %s""",
                (run_id, episode),
            )

    def fail(self, run_id: str, episode: int, error: str) -> None:
        """Release the unit back to the queue with the error recorded."""
        with self.cursor() as cur:
            cur.execute(
                """UPDATE work_units SET status = 'pending', lease_owner = NULL,
                   lease_expires_at = NULL, last_error = %s
                   WHERE run_id = %s AND episode = %s""",
                (error[:2000], run_id, episode),
            )

    def clear_episode(self, run_id: str, episode: int) -> None:
        """Drop any partial rows before a retry.

        A partially-completed episode restarts from day 1 rather than resuming.
        Rebuilding an LLM's mid-episode context after a crash risks a different
        prompt sequence than an uninterrupted run would have produced; a clean
        restart costs seconds and keeps every trajectory comparable.
        """
        with self.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_days WHERE run_id = %s AND episode = %s", (run_id, episode)
            )
            cur.execute(
                "DELETE FROM episodes WHERE run_id = %s AND episode = %s", (run_id, episode)
            )

    # ---- writes -----------------------------------------------------------

    def write_days(self, run_id: str, episode: int, rows) -> None:
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO agent_days (
                       run_id, episode, day, holding, age, wealth,
                       prompt_tokens, completion_tokens, thought_tokens, cache_hit,
                       raw_completion, thought_text,
                       parsed_belief, parsed_action, repair_used, error_class,
                       truth_m_t, truth_h_t, truth_pi_star, truth_q_loss, latency_ms)
                   VALUES (%(run_id)s, %(episode)s, %(day)s, %(holding)s, %(age)s, %(wealth)s,
                       %(prompt_tokens)s, %(completion_tokens)s, %(thought_tokens)s, %(cache_hit)s,
                       %(raw_completion)s, %(thought_text)s,
                       %(parsed_belief)s, %(parsed_action)s, %(repair_used)s,
                       %(error_class)s, %(truth_m_t)s, %(truth_h_t)s, %(truth_pi_star)s,
                       %(truth_q_loss)s, %(latency_ms)s)
                   ON CONFLICT (run_id, episode, day) DO NOTHING""",
                [dict(r, run_id=run_id, episode=episode) for r in rows],
            )

    def write_episode(self, run_id: str, summary: dict) -> None:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO episodes (
                       run_id, episode, true_exit_day, terminal_wealth, regret,
                       holdings, beliefs, switch_days, format_error_rate, empty_trunc_rate)
                   VALUES (%(run_id)s, %(episode)s, %(true_exit_day)s, %(terminal_wealth)s,
                       %(regret)s, %(holdings)s, %(beliefs)s, %(switch_days)s,
                       %(format_error_rate)s, %(empty_trunc_rate)s)
                   ON CONFLICT (run_id, episode) DO NOTHING""",
                dict(summary, run_id=run_id),
            )

    def record_spend(self, run_id: str, prompt: int, completion: int, thought: int,
                     usd: float) -> None:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO spend_ledger
                   (run_id, prompt_tokens, completion_tokens, thought_tokens, usd)
                   VALUES (%s, %s, %s, %s, %s)""",
                (run_id, prompt, completion, thought, usd),
            )

    # ---- reads for the dashboard -----------------------------------------

    def progress(self, run_id: str) -> dict:
        with self.cursor() as cur:
            cur.execute(
                """SELECT status, count(*) FROM work_units
                   WHERE run_id = %s GROUP BY status""",
                (run_id,),
            )
            return {status: count for status, count in cur.fetchall()}

    def list_runs(self) -> list:
        """Every cell this database holds, newest first, with its progress.

        The study is a grid of cells, not one run; the dashboard needs to name
        them before it can show any of them.
        """
        with self.cursor() as cur:
            cur.execute(
                """SELECT r.run_id, r.cell, r.status,
                          count(*) FILTER (WHERE w.status = 'done'),
                          count(*)
                   FROM runs r LEFT JOIN work_units w USING (run_id)
                   GROUP BY r.run_id, r.cell, r.status, r.started_at
                   ORDER BY r.started_at DESC"""
            )
            return [
                {"run_id": rid, "cell": cell or {}, "status": status,
                 "done": int(done or 0), "total": int(total or 0)}
                for rid, cell, status, done, total in cur.fetchall()
            ]

    def token_totals(self, run_id: str) -> dict:
        with self.cursor() as cur:
            cur.execute(
                """SELECT coalesce(sum(prompt_tokens), 0), coalesce(sum(completion_tokens), 0),
                          coalesce(sum(thought_tokens), 0), coalesce(sum(usd), 0)
                   FROM spend_ledger WHERE run_id = %s""",
                (run_id,),
            )
            prompt, completion, thought, usd = cur.fetchone()
        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "thought_tokens": int(thought),
            "total_tokens": int(prompt) + int(completion) + int(thought),
            "usd": float(usd),
        }

    def token_timeseries(self, run_id: str, bucket_seconds: int = 30):
        with self.cursor() as cur:
            cur.execute(
                """SELECT to_timestamp(floor(extract(epoch FROM ts) / %s) * %s) AS bucket,
                          sum(prompt_tokens), sum(completion_tokens), sum(thought_tokens)
                   FROM spend_ledger WHERE run_id = %s
                   GROUP BY bucket ORDER BY bucket""",
                (bucket_seconds, bucket_seconds, run_id),
            )
            return cur.fetchall()

    def error_counts(self, run_id: str) -> dict:
        with self.cursor() as cur:
            cur.execute(
                """SELECT error_class, count(*) FROM agent_days
                   WHERE run_id = %s GROUP BY error_class""",
                (run_id,),
            )
            return {cls: count for cls, count in cur.fetchall()}

    def thought_tokens_by_day(self, run_id: str):
        """Mean deliberation effort per day: does the model think harder when the
        horizon problem gets harder? Exploratory, added after the protocol."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT day, avg(thought_tokens), count(*) FROM agent_days
                   WHERE run_id = %s GROUP BY day ORDER BY day""",
                (run_id,),
            )
            return cur.fetchall()

    def active_day(self, run_id: str):
        """Mean episode day across in-flight work, for the hazard-ribbon marker."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT round(avg(day)) FROM agent_days
                   WHERE run_id = %s AND ts > now() - interval '2 minutes'""",
                (run_id,),
            )
            row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def switch_day_counts(self, run_id: str, horizon: int = 100):
        """Switches by day across completed episodes, for the trading chart."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT d, count(*) FROM episodes, unnest(switch_days) AS d
                   WHERE run_id = %s GROUP BY d ORDER BY d""",
                (run_id,),
            )
            rows = cur.fetchall()
        counts = [0] * (horizon + 1)
        for day, n in rows:
            if 0 <= day <= horizon:
                counts[day] = int(n)
        return counts

    def belief_error_by_day(self, run_id: str):
        """Mean stated belief per day, alongside the count behind it."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT day, avg(parsed_belief), count(parsed_belief) FROM agent_days
                   WHERE run_id = %s AND parsed_belief IS NOT NULL
                   GROUP BY day ORDER BY day""",
                (run_id,),
            )
            return cur.fetchall()

    def outcome_summary(self, run_id: str) -> dict:
        """Mean terminal wealth and regret over completed episodes."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT count(*), avg(terminal_wealth), avg(regret)
                   FROM episodes WHERE run_id = %s""",
                (run_id,),
            )
            n, wealth, regret = cur.fetchone()
        return {
            "episodes": int(n or 0),
            "mean_wealth": float(wealth) if wealth is not None else None,
            "regret": float(regret) if regret is not None else None,
        }

    # ---- decision log -----------------------------------------------------

    TRACE_COLUMNS = """episode, day, holding, age, wealth, parsed_belief, parsed_action,
                       truth_m_t, truth_h_t, truth_pi_star, truth_q_loss,
                       thought_tokens, error_class, raw_completion, thought_text"""

    def recent_trades(self, run_id: str, limit: int = 60):
        """Every day the agent actually changed position, newest first.

        A switch is where the holding differs from the previous day, so it is
        found with a window function rather than by trusting the episode summary:
        this way the model's own words for that day come back with it.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""WITH trace AS (
                        SELECT {self.TRACE_COLUMNS},
                               lag(holding) OVER (PARTITION BY episode ORDER BY day) AS came_from
                        FROM agent_days WHERE run_id = %s)
                    SELECT * FROM trace
                    WHERE came_from IS NULL OR came_from <> holding
                    ORDER BY episode DESC, day ASC
                    LIMIT %s""",
                (run_id, limit),
            )
            return cur.fetchall()

    def episode_trace(self, run_id: str, episode: int):
        """Every day of one episode, in order."""
        with self.cursor() as cur:
            cur.execute(
                f"""WITH trace AS (
                        SELECT {self.TRACE_COLUMNS},
                               lag(holding) OVER (PARTITION BY episode ORDER BY day) AS came_from
                        FROM agent_days WHERE run_id = %s AND episode = %s)
                    SELECT * FROM trace ORDER BY day""",
                (run_id, episode),
            )
            return cur.fetchall()

    def completed_episodes(self, run_id: str, limit: int = 40):
        """Finished episodes with enough summary to pick one worth opening."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT episode, true_exit_day, terminal_wealth, regret,
                          coalesce(array_length(switch_days, 1), 0)
                   FROM episodes WHERE run_id = %s
                   ORDER BY episode DESC LIMIT %s""",
                (run_id, limit),
            )
            return cur.fetchall()

    def costliest_decisions(self, run_id: str, limit: int = 12):
        """Days where departing from the optimal action cost the most wealth."""
        with self.cursor() as cur:
            cur.execute(
                f"""SELECT {self.TRACE_COLUMNS} FROM agent_days
                    WHERE run_id = %s AND truth_q_loss > 0
                    ORDER BY truth_q_loss DESC LIMIT %s""",
                (run_id, limit),
            )
            return cur.fetchall()

    def wealth_by_day(self, run_id: str):
        """The fleet equity curve: mean wealth held on each episode day.

        Averaged only over episodes still alive on that day, which is what the
        agents actually experienced; padding dead episodes forward would invent
        a flat tail that no agent ever traded through.
        """
        with self.cursor() as cur:
            cur.execute(
                """SELECT day, avg(wealth), min(wealth), max(wealth), count(*)
                   FROM agent_days WHERE run_id = %s GROUP BY day ORDER BY day""",
                (run_id,),
            )
            return cur.fetchall()

    def episode_records(self, run_id: str):
        with self.cursor() as cur:
            cur.execute(
                """SELECT episode, true_exit_day, terminal_wealth, holdings, beliefs, switch_days
                   FROM episodes WHERE run_id = %s ORDER BY episode""",
                (run_id,),
            )
            return cur.fetchall()
