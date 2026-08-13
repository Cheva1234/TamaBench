"""Deterministic, bounded V1.1 prompt construction."""

import json

from tamabench.schemas.observation import Observation


class ContextBuilder:
    MAX_RECENT_EVENTS: int = 6

    @classmethod
    def build_prompt(
        cls,
        observation: Observation,
        recent_event_descriptions: list[str],
        schema_mode: str = "raw_json",
    ) -> str:
        """Build one compact prompt without repeating mechanics or schema rules."""
        recent = recent_event_descriptions[-cls.MAX_RECENT_EVENTS:]
        events = "\n".join(f"- {event}" for event in recent) or "- none"
        state = json.dumps(observation.to_dict(), separators=(",", ":"))

        schema = (
            '{"action":"feed|play|clean|heal|sleep|wake|wait|work|buy|observe",'
            '"job_id":"...", "item":"food|medicine", "amount":1,'
            '"minutes":30, "hours":3|5|8, "prediction":{...}, "trace":{...}}'
        )
        if schema_mode != "raw_json":
            schema = "one provider-constrained JSON object using the TamaBench V1 schema"

        return f"""GOAL:
Keep the pet alive for 3 simulated days while managing money, energy, and supplies.

STATE:
{state}

ACTIONS:
feed consumes food and reduces hunger; clean restores cleanliness; heal consumes medicine;
play raises happiness; sleep(hours=3,5,8), wait(minutes), and work(job_id) block and fast-forward;
buy(item,amount) purchases supplies; wake and observe advance a short time.

WORLD RULES:
Hunger is a danger meter, not fullness: 0 means satisfied and 100 means starving.
Feeding consumes one food and lowers hunger by 35; do not feed only because the
number is high without checking whether food is available. Hunger rises by 18 per
simulated hour and is capped at 100. While hunger is above 85, health decreases
continuously by 0.2 per simulated minute (12 per hour), not just once. Before any
time-based action, estimate hunger at completion: a 3-hour sleep adds about 54
hunger, so sleeping at hunger 65 or higher reaches the danger zone. Sleeping only
recovers health when hunger is 50 or lower. If a time-based action can push hunger
above 85, feed first if food is available or choose a shorter action.
Cleanliness falls continuously; cleanliness < 20 damages health and increases
sickness risk. Sickness damages health. Agent energy limits care/work actions.
Health reaching 0 ends the episode.

RECENT EVENTS:
{events}

OUTPUT SCHEMA:
Return JSON only, with the existing V1 action, prediction, and trace fields:
{schema}
""".strip()
