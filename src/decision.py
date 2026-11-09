"""Turn model probabilities into an action, with safety rules.

| Top-1 confidence              | Action                                        |
|-------------------------------|-----------------------------------------------|
| >= CONF_ACCEPT (0.70)         | accept: speak it                              |
| CONF_MIN .. CONF_ACCEPT       | llm: let the LLM re-rank the top-3 by context |
| < CONF_MIN (0.40)             | repeat: say nothing, show the top-3 options   |
| any emergency >= EMERGENCY_MIN| alert caregivers immediately, then confirm    |

The emergency threshold is deliberately low: missing a real emergency is
worse than a false alarm.
"""
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class Decision:
    action: str                                   # "accept" | "llm" | "repeat"
    candidates: list = field(default_factory=list)  # [(phrase_id, prob)], best first
    phrase_id: str = None                         # set when action == "accept"
    alert_phrase_id: str = None                   # emergency phrase that raised an alert

    @property
    def confidence(self):
        return self.candidates[0][1] if self.candidates else 0.0


def top_k(probs, label_map, k=3):
    order = np.argsort(probs)[::-1][:k]
    return [(label_map[i], float(probs[i])) for i in order]


def decide(probs, label_map, phrases, conf_accept=config.CONF_ACCEPT,
           conf_min=config.CONF_MIN, emergency_min=config.EMERGENCY_MIN):
    probs = np.asarray(probs, dtype=float)
    candidates = top_k(probs, label_map, 3)

    alert = None
    for i in np.argsort(probs)[::-1]:
        if probs[i] < emergency_min:
            break
        if phrases.get(label_map[i], {}).get("emergency"):
            alert = label_map[i]
            break

    top_id, top_p = candidates[0]
    if top_p >= conf_accept:
        return Decision("accept", candidates, phrase_id=top_id, alert_phrase_id=alert)
    if top_p >= conf_min:
        return Decision("llm", candidates, alert_phrase_id=alert)
    return Decision("repeat", candidates, alert_phrase_id=alert)
