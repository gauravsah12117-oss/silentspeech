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


def combine_attempts(p_prev, p_new, floor=0.01):
    """Evidence from repeated tries: multiply the probabilities and renormalise.

    When the app says "not sure, please repeat", the next try should count
    together with the first. Two tries that agree on a phrase make it more
    likely; tries that disagree cancel out. Template probabilities may sum
    to less than 1 (the rest means "none of these"); that mass is kept as its
    own outcome so two "none of these" tries stay unsure. The floor stops one
    near-zero score from vetoing a phrase outright.
    """
    p_prev, p_new = np.asarray(p_prev, float), np.asarray(p_new, float)
    a = np.append(p_prev, max(0.0, 1.0 - p_prev.sum())) + floor
    b = np.append(p_new, max(0.0, 1.0 - p_new.sum())) + floor
    c = a * b
    return (c / c.sum())[:-1]


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
