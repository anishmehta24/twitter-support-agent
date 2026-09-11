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
R3 = HERE / "data" / "golden_r3.jsonl"   # human spot-check, see annotate.py


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

    # ---- agreement between rounds ----------------------------------------
    if not r2:
        print("\n(no round 2 yet - run `python annotate.py --round 2 --blind`)")
        return
    pairwise(r1, r2, "round 1", "round 2")

    # Round 3 exists when rounds 1 and 2 were both model passes and a human
    # spot-checked a subset. Human-vs-model on that subset is then the number
    # that matters; model-vs-model above is only label reliability.
    r3 = load(R3)
    if r3:
        pairwise(r3, r1, "round 3 (human)", "round 1")
        pairwise(r3, r2, "round 3 (human)", "round 2")
    else:
        print("\n(no round 3 - if rounds 1 and 2 are both model passes, run "
              "`python annotate.py --round 3 --blind` for a human spot-check)")


def who(rows: dict[str, dict]) -> str:
    ann = {r.get("annotator") or "human" for r in rows.values()}
    return "/".join(sorted(ann))


def pairwise(a: dict[str, dict], b: dict[str, dict], na: str, nb: str) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"\n(no overlap between {na} and {nb})")
        return {}
    print(f"\n=== {na} [{who(a)}]  vs  {nb} [{who(b)}]  over {len(shared)} messages ===\n")

    ia = [a[i]["intent"] for i in shared]
    ib = [b[i]["intent"] for i in shared]
    aa = [a[i]["action"] for i in shared]
    ab = [b[i]["action"] for i in shared]

    ik, ak = kappa(ia, ib), kappa(aa, ab)
    iagr = 100 * sum(x == y for x, y in zip(ia, ib)) / len(shared)
    aagr = 100 * sum(x == y for x, y in zip(aa, ab)) / len(shared)

    print(f"  intent : raw agreement {iagr:5.1f}%   kappa {ik:.3f}   {interpret(ik)}")
    print(f"  action : raw agreement {aagr:5.1f}%   kappa {ak:.3f}   {interpret(ak)}")

    disagreements = [(i, a[i]["intent"], b[i]["intent"]) for i in shared
                     if a[i]["intent"] != b[i]["intent"]]
    if disagreements:
        pairs = Counter(tuple(sorted((x, y))) for _, x, y in disagreements)
        print(f"\n  most confusable pairs ({len(disagreements)} disagreements):")
        for (x, y), n in pairs.most_common(6):
            print(f"    {n:>3}x  {x}  <->  {y}")
        print("\n  These pairs are where the taxonomy is genuinely ambiguous.")
        print("  Either merge them, or sharpen the guide and relabel.")
    return {"n": len(shared), "intent_kappa": ik, "action_kappa": ak,
            "intent_agreement": iagr, "action_agreement": aagr}


if __name__ == "__main__":
    utf8_console()
    main()
