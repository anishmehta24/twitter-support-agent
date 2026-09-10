# -*- coding: utf-8 -*-
"""Intent classifiers at three tiers, mirroring the reply tiers in reply.py.

    majority   predict the single most common gold intent for everything
    keyword    ordered regex rules, one pass, no learning
    llm        batched prompt against the taxonomy in label_intents.py

The first two exist so the LLM's macro-F1 has a denominator. On a corpus
where "where is my parcel" dominates, majority-class accuracy is not zero, and
a rule list written in twenty minutes is often embarrassingly close to a
model. If the LLM cannot clear both, the LLM is not earning its cost.

Every classifier returns the same Prediction so evaluate.py and agent.py do
not care which one is behind it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from label_intents import SYSTEM, TAXONOMY, build_prompt, parse_response
from sample_golden import detect_lang

HERE = Path(__file__).parent
LLM_CACHE = HERE / "data" / "predictions_llm.jsonl"


@dataclass
class Prediction:
    intent: str
    confidence: float
    language: str = "en"


class Classifier:
    name = "base"

    def predict(self, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError


class MajorityClassifier(Classifier):
    """Predicts one label for everything. Fit on the gold labels themselves -
    that is deliberate and stated: it is the floor, not a model."""
    name = "majority"

    def __init__(self, gold_intents: list[str] | None = None):
        self.label = Counter(gold_intents).most_common(1)[0][0] if gold_intents else "delivery_delayed"

    def predict(self, texts):
        return [Prediction(self.label, 1.0, detect_lang(t)) for t in texts]


# Ordered: first match wins. Order encodes the tie-breaks in ANNOTATION_GUIDE.md
# (Prime before payment; "marked delivered" before "still waiting"; a specific
# case before a general complaint). Deliberately crude - it is a baseline.
KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    ("account_access", re.compile(
        r"\b(log ?in|login|locked|password|verification|verify|otp|hacked|"
        r"suspended|two.step|2fa|sign in|signin|close my account)\b", re.I)),
    ("prime_membership", re.compile(r"\bprime\b", re.I)),
    ("payment_or_charge", re.compile(
        r"\b(charg\w*|payment|declin\w*|gift ?card|billed|debit|credit card|"
        r"amazon pay|double|twice)\b", re.I)),
    ("item_damaged_wrong_missing", re.compile(
        r"\b(damag\w*|broken|wrong (item|product|size|colour|color)|defective|"
        r"faulty|cracked|smashed|missing (item|part|piece)|not working|"
        r"doesn'?t work|stopped working|empty box|incomplete)\b", re.I)),
    ("refund_or_return", re.compile(r"\b(refund\w*|return\w*|money back)\b", re.I)),
    ("order_cancel_or_change", re.compile(
        r"\b(cancel\w*|change (my|the) (order|address|delivery)|modify|amend)\b", re.I)),
    ("delivery_not_received", re.compile(
        r"\b((says?|marked|shows?|showing|said|claims?) (it was |it'?s |as )?delivered|"
        r"delivered but|never (received|got|arrived|came)|not (received|here|arrived)|"
        r"(didn'?t|haven'?t|havent|did not|have not) (get|got|receive\w*|arrive\w*)|"
        r"stolen|left (it |the parcel )?(outside|at|with)|neighbou?r)\b", re.I)),
    ("delivery_delayed", re.compile(
        r"\b(late|delay\w*|still waiting|where'?s? (is )?my|wheres my|hasn'?t (arrived|shipped|come)|"
        r"not (shipped|dispatched|been delivered)|eta|expected|track\w*|waiting|"
        r"deliver\w*|shipping|shipment|package|parcel|order\w*)\b", re.I)),
    ("service_complaint_or_feedback", re.compile(
        r"\b(worst|terrible|disgust\w*|awful|horrible|useless|joke|thank\w*|"
        r"great|love|amazing|customer service|never again|ridiculous|shocking)\b", re.I)),
]


class KeywordClassifier(Classifier):
    name = "keyword"

    def predict(self, texts):
        out = []
        for t in texts:
            for intent, pat in KEYWORD_RULES:
                if pat.search(t):
                    out.append(Prediction(intent, 0.85, detect_lang(t)))
                    break
            else:
                out.append(Prediction("other", 0.40, detect_lang(t)))
        return out


class LLMClassifier(Classifier):
    """Batched, cached by message id. Same prompt as label_intents.py so the
    thing being evaluated is the thing that labelled the exploratory sample."""
    name = "llm"

    def __init__(self, client=None, batch: int = 10, cache: Path = LLM_CACHE):
        self._client = client
        self.batch = batch
        self.cache = cache
        self._mem: dict[str, dict] = {}
        if cache.exists():
            for line in cache.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self._mem[r["id"]] = r

    @property
    def client(self):
        if self._client is None:
            from llm import Client
            self._client = Client()
        return self._client

    def predict(self, texts, ids: list[str] | None = None):
        ids = ids or [f"anon-{i}" for i in range(len(texts))]
        todo = [(i, t) for i, t in zip(ids, texts) if i not in self._mem]
        if todo:
            with self.cache.open("a", encoding="utf-8") as sink:
                for s in range(0, len(todo), self.batch):
                    batch = todo[s:s + self.batch]
                    raw = self.client.complete(SYSTEM, build_prompt(batch), max_tokens=4096)
                    for r in parse_response(raw, batch):
                        r["model"] = self.client.model
                        self._mem[r["id"]] = r
                        if not r["id"].startswith("anon-"):
                            sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                    sink.flush()
                    print(f"  classified {min(s + self.batch, len(todo))}/{len(todo)}"
                          f"  ({self.client.usage})")
        out = []
        for i, t in zip(ids, texts):
            r = self._mem.get(i)
            if r is None:                       # model dropped it from the batch
                out.append(Prediction("other", 0.0, detect_lang(t)))
                continue
            lang = (r.get("language") or "").lower()[:2] or detect_lang(t)
            out.append(Prediction(r["intent"], float(r["confidence"]), lang))
        return out


CLASSIFIERS = {
    "majority": MajorityClassifier,
    "keyword": KeywordClassifier,
    "llm": LLMClassifier,
}


def make(name: str, gold_intents: list[str] | None = None, **kw) -> Classifier:
    if name == "majority":
        return MajorityClassifier(gold_intents)
    if name == "keyword":
        return KeywordClassifier()
    if name == "llm":
        return LLMClassifier(**kw)
    raise ValueError(f"unknown classifier {name!r}; choose from {list(CLASSIFIERS)}")


if __name__ == "__main__":
    import sys
    from llm import utf8_console
    utf8_console()
    msgs = sys.argv[1:] or [
        "my parcel says delivered but i never got it",
        "charged twice for prime membership this month",
        "cant log into my account, verification code never arrives",
        "the blender arrived with a cracked jug",
        "you people are the worst",
    ]
    kw = KeywordClassifier()
    for m, p in zip(msgs, kw.predict(msgs)):
        print(f"{p.intent:<32}{p.confidence:.2f}  {p.language}  {m[:60]}")
    print(f"\n{len(TAXONOMY)} intents; llm tier: python evaluate.py --classifier llm")
