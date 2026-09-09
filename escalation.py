# -*- coding: utf-8 -*-
"""Auto-handle vs escalate policy, with a stated reason for every decision.

The framing that matters: this is not a classification problem, it is a
decision under ASYMMETRIC COST. Wrongly auto-handling a fraud report is far
more expensive than needlessly escalating a "where is my parcel". So the
policy is deliberately conservative and escalation is the default; a message
must earn its way into auto-handling by clearing every gate.

Four gates, evaluated in order. Any failure escalates and names itself:

  1. HARD TRIGGER   - risk in the text (fraud, legal, safety, vulnerability).
                      Overrides everything, including high confidence.
  2. INTENT POLICY  - is this intent resolvable by a public tweet reply at all?
  3. CONFIDENCE     - is the classifier sure enough to act on?
  4. GROUNDING      - did retrieval find a real precedent to base a reply on?

Gate 4 is the one people forget. A confident label with no comparable
historical resolution means the agent would be improvising, which is exactly
when it should not be talking to a customer unsupervised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    AUTO = "auto_handle"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Gate 1: hard triggers. These override every other signal.
# Each is (name, pattern, why-it-matters).
# ---------------------------------------------------------------------------
HARD_TRIGGERS: list[tuple[str, re.Pattern, str]] = [
    # "not mine" was removed: it overwhelmingly matched misdelivered parcels
    # ("found a parcel at my door that's not mine"), which is a delivery intent,
    # not financial crime.
    ("fraud_or_unauthorised", re.compile(
        r"\b(fraud\w*|unauthorised|unauthorized|stolen card|card (was )?stolen|"
        r"didn'?t (order|authorise|authorize|make this (purchase|payment))|"
        r"someone (else )?(has )?(used|ordered|charged|bought)|"
        r"identity theft|scam(med|ming)?\b)", re.I),
     "possible financial crime; needs a human and an audit trail"),

    ("account_compromised", re.compile(
        r"\b(hack\w*|compromis\w*|breach\w*|someone (has|got) (into|access)|"
        r"can'?t (log ?in|access) .{0,25}(account|email)|account (locked|suspended|closed))\b", re.I),
     "account security; automated replies can aid an attacker"),

    ("legal_or_regulatory", re.compile(
        r"\b(lawyer|solicitor|legal action|sue|suing|court|ombudsman|"
        r"trading standards|small claims|gdpr|data protection|chargeback|"
        r"section 75|consumer rights)\b", re.I),
     "legal exposure; anything said becomes evidence"),

    # Amazon sells Fire TV / Fire Stick / Fire tablets, so a bare \bfire\b
    # matched product names far more often than hazards. Likewise "shock\w*"
    # matched "shockingly" (sentiment) and "recall" matched "I recall that".
    # Each now requires hazard context rather than a bare keyword.
    ("safety_or_harm", re.compile(
        r"\b(injur\w*|electrocut\w*|explod\w*|exploded|poison\w*|choking|"
        r"hospital|ambulance|paramedic)\b"
        r"|\b(caught|catches|set|started|burst into) fire\b"
        r"|\bon fire\b(?!\s*(tv|stick|tablet|hd|kids|cube))"
        r"|\bfire hazard\b|\bsmok(ing|e) (and|from|came)\b"
        r"|\belectric(al)? shock\b|\bgave me a shock\b"
        r"|\bburn(ed|t|ing)? (my|me|her|his|their)\b"
        r"|\b(product|safety) recall\b|\brecalled\b"
        r"|\bsevere\w* allerg\w*|\banaphyla\w*", re.I),
     "physical safety; product-safety escalation path exists for a reason"),

    ("vulnerability", re.compile(
        r"\b(disabled|disability|elderly|carer|hospice|terminal\w*|"
        r"bereave\w*|funeral|died|passed away|suicid\w*|kill myself|"
        r"depress\w*)\b", re.I),
     "vulnerable customer; requires human judgement and tone"),

    ("media_or_viral", re.compile(
        r"\b(journalist|reporter|press|bbc|watchdog|going viral|"
        r"@\w*news|tv show|newspaper)\b", re.I),
     "reputational risk; comms should own the response"),
]

# ---------------------------------------------------------------------------
# Gate 2: per-intent disposition.
#   AUTO_OK    - a grounded public reply can genuinely resolve or advance it
#   NEVER_AUTO - resolution requires account access, money movement, or judgement
# ---------------------------------------------------------------------------
AUTO_OK = {
    "delivery_delayed": "status/ETA guidance is a standard grounded reply",
    "delivery_not_received": "standard first-response is a documented checklist",
    "refund_or_return": "return process is publicly documented and stable",
    "order_cancel_or_change": "self-serve cancellation path is well documented",
    "service_complaint_or_feedback": "acknowledgement; praise needs no action",
}
NEVER_AUTO = {
    "payment_or_charge": "money movement and card data; needs verified account access",
    "prime_membership": "billing changes; needs verified account access",
    "account_access": "security-sensitive by definition",
    "item_damaged_wrong_missing": "requires claim/replacement decision and goodwill spend",
    "other": "unclassified by construction; never act on an unknown",
}

CONF_THRESHOLD = 0.75      # gate 3
GROUNDING_THRESHOLD = 0.60  # gate 4, cosine similarity to nearest precedent

# Cost model used to justify the thresholds above. Units are relative, not
# currency: what matters is the ratio, and that it is stated rather than implied.
COST_FALSE_AUTO = 10.0      # auto-handled something that needed a human
COST_FALSE_ESCALATE = 1.0   # escalated something the agent could have handled


@dataclass
class Decision:
    action: Action
    reason: str
    triggers: list[str] = field(default_factory=list)
    gate: str = ""

    def __str__(self) -> str:
        return f"{self.action.value}: {self.reason}"


def decide(
    intent: str,
    confidence: float,
    text: str,
    grounding: float | None = None,
    multi_intent: bool = False,
    language: str = "en",
) -> Decision:
    """Return an auto/escalate decision with a human-readable reason."""
    # Gate 1 - hard triggers, checked first and unconditionally.
    hits = [(name, why) for name, pat, why in HARD_TRIGGERS if pat.search(text)]
    if hits:
        names = [n for n, _ in hits]
        return Decision(
            Action.ESCALATE,
            f"risk signal ({', '.join(names)}): {hits[0][1]}",
            triggers=names, gate="hard_trigger",
        )

    # Gate 2 - is the intent resolvable in this channel at all?
    if intent in NEVER_AUTO:
        return Decision(
            Action.ESCALATE, f"intent '{intent}' always escalates: {NEVER_AUTO[intent]}",
            gate="intent_policy",
        )
    if intent not in AUTO_OK:
        return Decision(
            Action.ESCALATE, f"unknown intent '{intent}'; refusing to act on an unmapped label",
            gate="intent_policy",
        )

    # Ambiguity: a message carrying several problems cannot be resolved by one
    # templated reply, however confident the primary label is.
    if multi_intent:
        return Decision(
            Action.ESCALATE,
            "multiple distinct problems in one message; a single reply would drop one",
            gate="ambiguity",
        )

    # Language: only auto-handle where replies can be grounded in same-language
    # precedent. ~12% of this corpus is not English.
    if language and language != "en":
        return Decision(
            Action.ESCALATE,
            f"non-English message ({language}); no grounded precedent pool in that language",
            gate="language",
        )

    # Gate 3 - classifier confidence.
    if confidence < CONF_THRESHOLD:
        return Decision(
            Action.ESCALATE,
            f"low intent confidence {confidence:.2f} < {CONF_THRESHOLD:.2f}",
            gate="confidence",
        )

    # Gate 4 - grounding. No precedent means the reply would be improvised.
    if grounding is not None and grounding < GROUNDING_THRESHOLD:
        return Decision(
            Action.ESCALATE,
            f"weak grounding {grounding:.2f} < {GROUNDING_THRESHOLD:.2f}; "
            "no comparable historical resolution to base a reply on",
            gate="grounding",
        )

    return Decision(
        Action.AUTO,
        f"intent '{intent}' ({AUTO_OK[intent]}); confidence {confidence:.2f}"
        + (f", grounding {grounding:.2f}" if grounding is not None else ""),
        gate="passed_all",
    )


def expected_cost(false_autos: int, false_escalations: int) -> float:
    """Single number to compare operating points. State the ratio in the report."""
    return false_autos * COST_FALSE_AUTO + false_escalations * COST_FALSE_ESCALATE


if __name__ == "__main__":
    cases = [
        ("delivery_delayed", 0.92, "where is my parcel? said 2 day delivery, its been 5", 0.81, False, "en"),
        ("delivery_not_received", 0.88, "tracking says delivered but nothing here", 0.74, False, "en"),
        ("payment_or_charge", 0.95, "charged twice for the same order", 0.90, False, "en"),
        ("delivery_delayed", 0.91, "my parcel is late AND someone used my card to order it", 0.85, False, "en"),
        ("delivery_delayed", 0.55, "wheres my stuff", 0.70, False, "en"),
        ("refund_or_return", 0.90, "how do i return a toaster", 0.31, False, "en"),
        ("delivery_delayed", 0.93, "mi pedido no ha llegado todavia", 0.80, False, "es"),
        ("service_complaint_or_feedback", 0.89, "Sasha was super helpful, thanks Amazon!", 0.77, False, "en"),
        ("delivery_delayed", 0.94, "parcel late, my elderly mother has been waiting in all day", 0.88, False, "en"),
    ]
    print(f"{'INTENT':<32}{'ACTION':<13}{'GATE':<16}REASON")
    print("-" * 118)
    for intent, conf, text, ground, multi, lang in cases:
        d = decide(intent, conf, text, ground, multi, lang)
        print(f"{intent:<32}{d.action.value:<13}{d.gate:<16}{d.reason[:62]}")
