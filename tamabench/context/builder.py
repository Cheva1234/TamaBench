"""Compact Context Builder for TamaBench V1.

Enforces a fixed token budget (~2K-3K tokens max) using sliding windows
over recent events to guarantee ZERO CONTEXT GROWTH as simulation horizon extends to weeks.
"""

import json
from typing import Any
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
        """Builds a token-efficient prompt string within fixed budget constraints."""
        obs_dict = observation.to_dict()

        # Sliding window over latest events
        recent = recent_event_descriptions[-cls.MAX_RECENT_EVENTS:] if recent_event_descriptions else []
        events_formatted = "\n".join(f"- {e}" for e in recent) if recent else "None"

        # System Instructions & Schema Guidance
        if schema_mode == "raw_json":
            schema_guidance = """YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT matching this exact schema:
{
  "action": "<feed|play|clean|heal|sleep|wake|wait|work|buy|observe>",
  "job_id": "<cafe_shift|delivery|freelance>",  // Required ONLY if action is 'work'
  "item": "<food|medicine>",                  // Required ONLY if action is 'buy'
  "amount": 1,                                  // Optional quantity for buy (e.g. 1, 3, 5)
  "hours": 8,                                   // Duration for sleep: must specify 3, 5, or 8
  "minutes": 30,                                // Optional duration for wait (e.g. 30, 60)
  "prediction": {                               // Mandatory structured prediction
    "pet_safe_until_completion": true,
    "expected_money_after": 100,
    "expected_hunger_after": 40.0
  },
  "trace": {                                    // Mandatory decision trace
    "situation_summary": "<brief situation>",
    "current_priority": "<primary priority>",
    "chosen_action": "<action>",
    "decision_rationale": "<reasoning>",
    "confidence": 0.95
  }
}
Do NOT wrap output in markdown. Output JSON string directly."""
        else:
            schema_guidance = "Provide structured action object matching requested schema."

        prompt = f"""[TAMABENCH AGENT MISSION]
You are an autonomous AI Agent managing your energy, earning money through jobs, purchasing supplies, and caring for your Tamagotchi pet.

[ENVIRONMENT MECHANICS & SPECIFICATIONS]
- Survival Boundary:
  • If pet health reaches 0.0, THE PET DIES IMMEDIATELY AND THE BENCHMARK EPISODE FAILS.
- Pet Stat Dynamics:
  • Health (0.0 to 100.0): Maximum 100.0. Drains rapidly when hunger is critical, cleanliness is low, or pet is sick!
  • Hunger (0.0 to 100.0): 0.0 = Satiated (Full), 100.0 = Starving. Increases continuously by +0.30/min (+18.0/hr).
    HAZARD: If hunger > 85.0, pet enters Critical Hunger state and health decays rapidly (-0.2/min = -12 health/hr)!
  • Cleanliness (0.0 to 100.0): 100.0 = Clean, 0.0 = Dirty. Decreases continuously by -0.15/min (-9.0/hr).
    HAZARD: If cleanliness < 20.0, health decays (-0.1/min = -6 health/hr) and sickness risk increases!
  • Sickness: If pet becomes sick, health decays severely (-0.3/min = -18 health/hr)!
  • Energy (0.0 to 100.0): Decreases by -0.20/min when awake. Recovers during 'sleep' (+0.5/min).
  • Cumulative Decay: Critical hunger, low cleanliness, and sickness penalties STACK together to kill the pet in just a few hours if unmanaged!
- Agent Energy Dynamics:
  • Maximum 100.0. Drains when awake (-0.12/min) and when working/playing/cleaning.
  • If agent energy is too low, agent CANNOT work or perform care actions!
  • 'sleep' or 'wait' recovers agent energy (+0.4/min).

[ACTION SPECIFICATIONS]
- Care Actions:
  • 'feed': Consumes 1 food from inventory. Reduces hunger by 35.0 (towards 0.0). Duration: 2m.
  • 'clean': Bathes pet. Restores cleanliness to 100.0%. Cost: 5 agent energy. Duration: 10m.
  • 'heal': Consumes 1 medicine from inventory. Cures sickness (if sick) and restores +15.0 health. Can be used anytime health is below 100 — not only when sick. Duration: 5m.
  • 'play': Restores happiness (+20.0). Cost: 10 agent energy, 10 pet energy. Duration: 15m.
- Recovery Actions:
  • 'sleep': Fast-forwards requested 'hours' (specify 'hours': 3, 5, or 8). Recovers agent (+0.4/min) and pet energy (+0.5/min). Also recovers health (+0.05/min) if pet is healthy and hunger <= 50.
  • 'wake': Wakes sleeping pet. Duration: 1m.
  • 'wait': Fast-forwards specified 'minutes' (default 60m, max 120m). Recovers agent energy (+0.4/min).
- Economy Actions:
  • 'work': Performs specified 'job_id' to earn money. Jobs: 'cafe_shift' (60m, +$25, cost 20 energy), 'delivery' (120m, +$55, cost 40 energy), 'freelance' (30m, +$10, cost 12 energy).
  • 'buy': Purchases specified 'item' ('food' for $30/unit, 'medicine' for $75/unit). Accepts 'amount' for bulk purchasing (e.g., 'amount': 5 buys 5 items in 1 step).
  • 'observe': Inspects current state. Duration: 1m.

[PRECONDITIONS]
- Cannot feed, clean, or heal pet while pet is sleeping.
- Cannot perform actions if required agent energy or inventory item is insufficient.

[CURRENT WORLD STATE]
{json.dumps(obs_dict, indent=2)}

[RECENT EVENT HISTORY (LAST 6)]
{events_formatted}

[RESPONSE REQUIREMENT]
{schema_guidance}
"""
        return prompt
