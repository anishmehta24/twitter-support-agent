# -*- coding: utf-8 -*-
"""Sample the golden evaluation set.

The central tension: a uniform random sample of 250 messages is representative
but nearly useless. Hard triggers are 2.1% of the corpus, so you would draw ~5
of them and could not evaluate the escalation policy at all. Non-English is
~12%, so ~30 - barely enough to say anything per language.

So we oversample the rare-but-important strata. The cost is that raw accuracy
over this set is NOT population accuracy. To recover it, every row carries:

    stratum  - which disjoint bucket it came from
    weight   - population_share / sample_share

Report BOTH: per-stratum metrics (how well does it do on fraud cases?) and
weight-corrected overall metrics (what would we see in production?). Quoting
only the unweighted overall number would be misleading, and that is precisely
what the report's mandatory section is asking about.

Strata are disjoint and assigned by priority, so a French fraud report counts
once, as hard_trigger.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from escalation import HARD_TRIGGERS

HERE = Path(__file__).parent
SRC = HERE / "data" / "amazon_clusters.tsv"
OUT = HERE / "data" / "golden_pool.jsonl"
SEED = 20260910
TARGET = 250

# Allocation. Deliberately not proportional - see module docstring.
ALLOCATION = {
    "hard_trigger": 40,     # 2.1% of corpus; without this, escalation is unevaluable
    "non_english": 30,      # ~12% of corpus, spread over 3+ languages
    "short_or_fragment": 30,  # the "I tried making this happen. Help!" residual
    "core": 150,            # representative remainder
}

# Cheap, explainable language detection: function-word hits beat a dependency
# here, and you can defend it in an interview. Only needs to separate the three
# languages the clustering surfaced (fr/es/de) from English.
LANG_MARKERS = {
    "fr": r"\b(je|jai|j'ai|bonjour|est|pas|vous|une|mon|ma|commande|colis|merci|pour|avec|plus)\b",
    "es": r"\b(que|el|la|los|mi|por|hola|pedido|una|se|para|con|como|no he|todavia|todavía)\b",
    "de": r"\b(ich|nicht|das|ist|und|die|der|es|kann|mit|habe|bestellung|paket|auch|noch)\b",
}
LANG_RE = {k: re.compile(v, re.I) for k, v in LANG_MARKERS.items()}


def detect_lang(text: str) -> str:
    scores = {k: len(p.findall(text)) for k, p in LANG_RE.items()}
    best = max(scores, key=lambda k: scores[k])
    # Require 2+ marker hits: one stray "no" or "die" should not flip a tweet.
    return best if scores[best] >= 2 else "en"


def has_hard_trigger(text: str) -> bool:
    return any(p.search(text) for _n, p, _w in HARD_TRIGGERS)


def stratum_of(text: str) -> str:
    """Disjoint, priority-ordered."""
    if has_hard_trigger(text):
        return "hard_trigger"
    if detect_lang(text) != "en":
        return "non_english"
    if len(text) < 60 or text.count(" ") < 8:
        return "short_or_fragment"
    return "core"


def main() -> None:
    rows = []
    with SRC.open("r", encoding="utf-8", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter="\t")):
            rows.append({"id": f"t{i}", "cluster": r["cluster"], "text": r["text"]})

    for r in rows:
        r["stratum"] = stratum_of(r["text"])
        r["language"] = detect_lang(r["text"])

    pop = Counter(r["stratum"] for r in rows)
    n_pop = len(rows)
    print(f"population: {n_pop:,} openers\n")
    print(f"{'STRATUM':<20}{'POP':>8}{'POP%':>8}{'SAMPLE':>8}{'SAMP%':>8}{'WEIGHT':>9}")
    print("-" * 61)

    random.seed(SEED)
    by_stratum: dict[str, list[dict]] = {}
    for s in ALLOCATION:
        by_stratum[s] = [r for r in rows if r["stratum"] == s]

    sample: list[dict] = []
    for s, want in ALLOCATION.items():
        pool = by_stratum[s]
        take = min(want, len(pool))
        # Within a stratum, spread across clusters so one topic cannot dominate.
        pool_sorted = sorted(pool, key=lambda r: (r["cluster"], r["id"]))
        picked = random.sample(pool_sorted, take)
        pop_share = pop[s] / n_pop
        samp_share = take / TARGET
        weight = pop_share / samp_share if samp_share else 0.0
        print(f"{s:<20}{pop[s]:>8,}{100*pop_share:>7.1f}%{take:>8}"
              f"{100*samp_share:>7.1f}%{weight:>9.3f}")
        for r in picked:
            r = dict(r)
            r["weight"] = round(weight, 4)
            sample.append(r)

    random.shuffle(sample)  # so the annotator does not label one stratum in a block

    with OUT.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(sample):
            r["order"] = i
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("-" * 61)
    print(f"{'TOTAL':<20}{n_pop:>8,}{100.0:>7.1f}%{len(sample):>8}{100.0:>7.1f}%")
    print(f"\nlanguages in sample: "
          + ", ".join(f"{k}={v}" for k, v in
                      Counter(r['language'] for r in sample).most_common()))
    print(f"written: {OUT}")
    print("\nNext: python annotate.py           (round 1)")
    print("      python annotate.py --round 2 --blind   (after a day)")
    print("      python agreement.py            (self-agreement kappa)")


if __name__ == "__main__":
    main()
