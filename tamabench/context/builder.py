"""Deterministic, bounded V1.1 prompt construction."""

import json

from tamabench.schemas.observation import Observation
from tamabench.env.scenarios import get_config_for_scenario


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
        config = get_config_for_scenario(observation.scenario_id)
        simulated_days = config.max_simulated_minutes / 1440
        day_label = f"{simulated_days:g} simulated day{'s' if simulated_days != 1 else ''}"

        schema = (
            '{"action":"feed|play|clean|heal|sleep|wake|wait|work|buy|observe",'
            '"job_id":"...", "item":"food|medicine", "amount":1,'
            '"minutes":30, "hours":3|5|8, "prediction":{...}, "trace":{...}}'
        )
        if schema_mode != "raw_json":
            schema = "one provider-constrained JSON object using the TamaBench V1 schema"

        return f"""GOAL:
Keep the pet alive for {day_label} while managing money, energy, and supplies.
Difficulty: {config.name}

STATE:
{state}

ACTIONS:
feed consumes food and increases hunger/fullness; clean restores cleanliness; heal consumes medicine;
play raises happiness; sleep(hours=3,5,8), wait(minutes), and work(job_id) block and fast-forward;
buy(item,amount) purchases supplies; wake and observe advance a short time.

SMALL-MODEL OUTPUT RULE:
The prediction and trace fields are optional. Omit them unless you can follow
their exact nested schema. These are valid minimal outputs:
{{"action":"feed"}}
{{"action":"wait","minutes":60}}
{{"action":"sleep","hours":3}}
Never put the full observation inside prediction, and never make trace a string.

WORLD RULES:
Hunger is a fullness meter, not a starvation score: 0 means starving and 100 means fully fed.
Feeding increases hunger by 35 and consumes one food; do not feed when fullness is
already high (70 or more) because food is limited. Hunger decreases by 18 per simulated hour
and is capped at 0. While hunger is below 15, health decreases continuously by
0.2 per simulated minute (12 per hour), not just once. Before any time-based
action, estimate hunger at completion: a 3-hour sleep removes about 54 hunger, so
sleeping at hunger 50 or lower reaches the danger zone. Sleeping only recovers
health when hunger is 50 or higher. If a time-based action can push hunger below
15, feed first when fullness is low or choose a shorter action.
Cleanliness falls continuously; cleanliness < 20 damages health and increases
sickness risk. Sickness damages health. Agent energy limits care/work actions.
Health reaching 0 ends the episode.

RECENT EVENTS:
{events}

OUTPUT SCHEMA:
Return JSON only, with the existing V1 action, prediction, and trace fields:
{schema}
""".strip()
