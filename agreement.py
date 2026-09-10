# -*- coding: utf-8 -*-
"""Self-agreement between annotation rounds, and golden-set health.

Why this exists: a solo annotator cannot report inter-annotator agreement, so
the honest substitute is intra-annotator agreement - label the same messages
twice, blind, days apart, and measure how often you agree with yourself.

That number is the ceiling on what any model can be shown to achieve. If your
own kappa is 0.72, a classifier scoring 0.90 against your labels is not
measuring what you think it is.

Cohen's kappa is used rather than raw agreement because raw agreement is
inflated by class imbalance: always guessing the majority intent would look
respectable on percentage agreement and score ~0 on kappa.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from llm import utf8_console

HERE = Path(__file__).parent
R1 = HERE / "data" / "golden_r1.jsonl"
R2 = HERE / "data" / "golden_r2.jsonl"


def load(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l)
            for l in p.open(encoding="utf-8") if l.strip()}


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two label sequences over the same items."""
    if not a:
        return float("nan")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret(k: float) -> str:
    if k != k:
        return ""
    if k < 0.20:
        return "slight - the taxonomy is not usable as written"
    if k < 0.40:
        return "fair - categories overlap badly, revise before trusting metrics"
    if k < 0.60:
        return "moderate - usable but report it as a caveat"
    if k < 0.80:
        return "substantial - normal for subjective support labelling"
    return "almost perfect"


def main() -> None:
    r1, r2 = load(R1), load(R2)
    if not r1:
        raise SystemExit("no round 1 labels - run `python annotate.py`")

    # ---- golden set health (round 1 alone) --------------------------------
    rows = list(r1.values())
    print(f"=== round 1: {len(rows)} labelled ===\n")

    print("intent distribution (raw vs weight-corrected):")
    raw = Counter(r["intent"] for r in rows)
    wsum: dict[str, float] = defaultdict(float)
    for r in rows:
        wsum[r["intent"]] += r.get("weight", 1.0)
    tw = sum(wsum.values()) or 1.0
    for k, v in raw.most_common():
        print(f"  {k:<32}{v:>5} ({100*v/len(rows):>5.1f}%)   "
              f"weighted {100*wsum[k]/tw:>5.1f}%")

    acts = Counter(r["action"] for r in rows)
    wa: dict[str, float] = defaultdict(float)
    for r in rows:
        wa[r["action"]] += r.get("weight", 1.0)
    print("\naction distribution:")
    for k, v in acts.most_common():
        print(f"  {k:<32}{v:>5} ({100*v/len(rows):>5.1f}%)   "
              f"weighted {100*wa[k]/tw:>5.1f}%")

    multi = sum(1 for r in rows if r.get("multi"))
    unsure = sum(1 for r in rows if r.get("unsure"))
    print(f"\nmulti-intent : {multi:>4} ({100*multi/len(rows):.1f}%) "
          "<- hard ceiling on single-label accuracy")
    print(f"unsure       : {unsure:>4} ({100*unsure/len(rows):.1f}%) "
          "<- your own uncertainty, report it")

    # ---- self-agreement ---------------------------------------------------
    if not r2:
        print("\n(no round 2 yet - run `python annotate.py --round 2 --blind`)")
        return

    shared = sorted(set(r1) & set(r2))
    print(f"\n=== self-agreement over {len(shared)} doubly-labelled messages ===\n")

    ia = [r1[i]["intent"] for i in shared]
    ib = [r2[i]["intent"] for i in shared]
    aa = [r1[i]["action"] for i in shared]
    ab = [r2[i]["action"] for i in shared]

    ik, ak = kappa(ia, ib), kappa(aa, ab)
    iagr = 100 * sum(x == y for x, y in zip(ia, ib)) / len(shared)
    aagr = 100 * sum(x == y for x, y in zip(aa, ab)) / len(shared)

    print(f"  intent : raw agreement {iagr:5.1f}%   kappa {ik:.3f}   {interpret(ik)}")
    print(f"  action : raw agreement {aagr:5.1f}%   kappa {ak:.3f}   {interpret(ak)}")

    disagreements = [(i, r1[i]["intent"], r2[i]["intent"]) for i in shared
                     if r1[i]["intent"] != r2[i]["intent"]]
    if disagreements:
        pairs = Counter(tuple(sorted((a, b))) for _, a, b in disagreements)
        print(f"\n  most confusable pairs ({len(disagreements)} disagreements):")
        for (a, b), n in pairs.most_common(6):
            print(f"    {n:>3}x  {a}  <->  {b}")
        print("\n  These pairs are where your taxonomy is genuinely ambiguous.")
        print("  Either merge them, or sharpen the guide and relabel.")


if __name__ == "__main__":
    utf8_console()
    main()
