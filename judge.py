# -*- coding: utf-8 -*-
"""LLM-as-judge for reply quality, and the human protocol that validates it.

A judge score means nothing on its own. The assignment is explicit: include
evidence of how well the judge agrees with a human. So this module has two
halves that share one rubric:

    python judge.py --human --n 40      you score a blind, shuffled subset
    python judge.py --agreement         judge vs you: kappa per criterion

Rubric. Four binary checks plus one holistic score. Binary because a human
scoring 40 replies at 1-5 granularity per criterion produces noise, not
signal, and because kappa on a binary is interpretable:

    grounded    consistent with the precedents; invents no policy, amount, date
    safe        does not request personal details publicly; makes no promise
                a public reply cannot keep
    actionable  offers a concrete next step that advances THIS case
    on_tone     reads like the brand: brief, apologetic where due, no filler
    overall     1-5, holistic - would a support lead let this go out?

Two things the judge cannot see, stated so nobody mistakes the score for
resolution quality: (1) whether the precedent itself was a good answer - a
reply grounded in a bad precedent scores "grounded"; (2) whether the case is
actually resolved - a polite redirect can score 4/5 while resolving nothing.

The human view hides the tier. If you can tell trivial from LLM from the
layout you will score the layout.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from agreement import interpret, kappa
from llm import utf8_console

HERE = Path(__file__).parent
DATA = HERE / "data"
HUMAN = DATA / "judge_human.jsonl"
CRITERIA = ["grounded", "safe", "actionable", "on_tone"]

JUDGE_SYSTEM = (
    "You are auditing draft replies from a brand's public Twitter support "
    "account. For each item you see the customer's message, the precedents "
    "(how the brand historically answered similar messages), and a draft reply.\n\n"
    "Score each draft on four binary criteria and one overall score:\n"
    "- grounded (0/1): consistent with the precedents; invents no policy, amount, "
    "date, or process not supported by them. A generic apology with no claims is grounded.\n"
    "- safe (0/1): does not ask for order numbers, addresses, card or account "
    "details in public; does not promise a refund/replacement it cannot guarantee.\n"
    "- actionable (0/1): gives a concrete next step that advances THIS customer's "
    "case. 'Contact us' alone is NOT actionable unless the precedents show that is the only path.\n"
    "- on_tone (0/1): brief, apologetic where warranted, no filler, no hashtags, "
    "under 280 characters, reads like the brand.\n"
    "- overall (1-5): would a support lead let this go out unedited? "
    "5 = yes, 3 = needs a small edit, 1 = harmful or useless.\n\n"
    "Be strict. Score the reply, not the customer. Ignore which method produced it."
)


def build_judge_prompt(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"### ITEM {i}")
        lines.append(f"CUSTOMER: {it['text']}")
        lines.append("PRECEDENTS:")
        for p in it.get("precedents", [])[:3]:
            lines.append(f"  - customer: {p['customer'][:200]}")
            lines.append(f"    brand:    {p['reply'][:200]}")
        lines.append(f"DRAFT REPLY: {it['reply']}\n")
    # No example numbers here: a small model will copy them verbatim for
    # every item. Likewise the rationale is described, not templated.
    lines.append(
        f"Return ONLY a JSON array with exactly {len(items)} objects, one per "
        "item, in order. Keys: n (item number), grounded, safe, actionable, "
        "on_tone (each 0 or 1), overall (integer 1-5), rationale (one short "
        "sentence explaining the overall score)."
    )
    return "\n".join(lines)


def parse_judge(raw: str, n: int) -> dict[int, dict]:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("["), s.rfind("]")
    if a == -1 or b == -1:
        raise ValueError(f"no JSON array in judge response: {raw[:200]}")
    out = {}
    for obj in json.loads(s[a:b + 1]):
        k = int(obj.get("n", 0))
        if not 1 <= k <= n:
            continue
        rec = {c: int(bool(int(obj.get(c, 0) or 0))) for c in CRITERIA}
        rec["overall"] = max(1, min(5, int(obj.get("overall", 1) or 1)))
        rec["rationale"] = str(obj.get("rationale", ""))[:160]
        out[k] = rec
    return out


class Judge:
    """Batched and cached per (tier, id). Judge and generator should not be the
    same model where you can help it; with one local model they are, and that
    is a caveat for the report, not something to hide."""

    def __init__(self, client=None, batch: int = 5):
        self._client = client
        self.batch = batch

    @property
    def client(self):
        if self._client is None:
            from llm import Client
            self._client = Client()
        return self._client

    def score(self, items: list[dict], cache: Path) -> dict[str, dict]:
        done: dict[str, dict] = {}
        if cache.exists():
            for line in cache.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
        todo = [it for it in items if it["id"] not in done]
        if todo:
            with cache.open("a", encoding="utf-8") as sink:
                for s in range(0, len(todo), self.batch):
                    batch = todo[s:s + self.batch]
                    raw = self.client.complete(JUDGE_SYSTEM, build_judge_prompt(batch),
                                               max_tokens=2048)
                    scored = parse_judge(raw, len(batch))
                    # Small models often answer only the first item of a batch.
                    # Re-ask for the stragglers one at a time rather than lose them.
                    for k in range(1, len(batch) + 1):
                        if k not in scored:
                            one = parse_judge(self.client.complete(
                                JUDGE_SYSTEM, build_judge_prompt([batch[k - 1]]),
                                max_tokens=512), 1)
                            if 1 in one:
                                scored[k] = one[1]
                    for k, rec in scored.items():
                        rec = {"id": batch[k - 1]["id"], "model": self.client.model, **rec}
                        done[rec["id"]] = rec
                        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                    print(f"  judged {min(s + self.batch, len(todo))}/{len(todo)}"
                          f"  ({self.client.usage})")
        return done


# ---------------------------------------------------------------------------
# Human protocol
# ---------------------------------------------------------------------------
def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def human(n: int, seed: int = 7) -> None:
    """Score a blind, shuffled subset drawn evenly across tiers."""
    tiers = sorted(p.stem.replace("replies_", "") for p in DATA.glob("replies_*.jsonl"))
    if not tiers:
        raise SystemExit("no replies yet - run `python evaluate.py` first")
    pool = []
    for t in tiers:
        for r in _load(DATA / f"replies_{t}.jsonl"):
            pool.append({"tier": t, **r})
    random.seed(seed)
    random.shuffle(pool)
    # Even split across tiers, then shuffle again so tier order is not a cue.
    per = max(1, n // len(tiers))
    chosen = []
    for t in tiers:
        chosen += [r for r in pool if r["tier"] == t][:per]
    random.shuffle(chosen)

    done = {(r["tier"], r["id"]) for r in _load(HUMAN)}
    todo = [r for r in chosen if (r["tier"], r["id"]) not in done]
    print(f"\n{len(done)} scored, {len(todo)} to go.  For each: four 0/1 flags "
          f"then overall 1-5, e.g.  '1 1 0 1 3'   (q quits)")
    print("criteria: grounded  safe  actionable  on_tone  |  overall\n")

    with HUMAN.open("a", encoding="utf-8") as sink:
        for i, r in enumerate(todo, 1):
            print("=" * 78)
            print(f"[{i}/{len(todo)}]  CUSTOMER: {r['text']}")
            print("PRECEDENTS:")
            for p in r.get("precedents", [])[:3]:
                print(f"  - {p['reply'][:140]}")
            print(f"\nDRAFT: {r['reply']}\n")
            while True:
                raw = input("> ").strip().lower()
                if raw == "q":
                    return
                parts = raw.split()
                if len(parts) != 5 or not all(x.isdigit() for x in parts):
                    print("  need 5 numbers: g s a t overall")
                    continue
                g, s_, a, t, o = (int(x) for x in parts)
                if not all(x in (0, 1) for x in (g, s_, a, t)) or not 1 <= o <= 5:
                    print("  flags are 0/1, overall is 1-5")
                    continue
                sink.write(json.dumps({
                    "id": r["id"], "tier": r["tier"], "grounded": g, "safe": s_,
                    "actionable": a, "on_tone": t, "overall": o,
                }, ensure_ascii=False) + "\n")
                sink.flush()
                break
    print(f"\ndone -> {HUMAN}\nNow: python judge.py --agreement")


def agreement() -> dict:
    """Judge vs human on the rows both have scored. Returns a dict for results.json."""
    hum = _load(HUMAN)
    if not hum:
        raise SystemExit("no human scores - run `python judge.py --human --n 40`")
    judged: dict[tuple[str, str], dict] = {}
    for p in DATA.glob("judge_*.jsonl"):
        if p.name == HUMAN.name:
            continue
        tier = p.stem.replace("judge_", "")
        for r in _load(p):
            judged[(tier, r["id"])] = r
    pairs = [(h, judged[(h["tier"], h["id"])]) for h in hum
             if (h["tier"], h["id"]) in judged]
    if not pairs:
        raise SystemExit("no overlap between human and judge scores yet")

    out = {"n": len(pairs), "criteria": {}}
    print(f"\n=== judge vs human over {len(pairs)} replies ===\n")
    print(f"{'criterion':<12}{'agree%':>8}{'kappa':>8}  interpretation")
    for c in CRITERIA:
        a = [str(h[c]) for h, _ in pairs]
        b = [str(j[c]) for _, j in pairs]
        k = kappa(a, b)
        agr = 100 * sum(x == y for x, y in zip(a, b)) / len(pairs)
        out["criteria"][c] = {"agreement": agr, "kappa": k}
        print(f"{c:<12}{agr:>7.1f}%{k:>8.3f}  {interpret(k)}")

    ho = [h["overall"] for h, _ in pairs]
    jo = [j["overall"] for _, j in pairs]
    exact = 100 * sum(x == y for x, y in zip(ho, jo)) / len(pairs)
    within1 = 100 * sum(abs(x - y) <= 1 for x, y in zip(ho, jo)) / len(pairs)
    rho = spearman(ho, jo)
    bias = sum(j - h for h, j in zip(ho, jo)) / len(pairs)
    out["overall"] = {"exact": exact, "within_1": within1, "spearman": rho,
                      "judge_minus_human": bias}
    print(f"\noverall 1-5 : exact {exact:.1f}%   within-1 {within1:.1f}%   "
          f"spearman {rho:.3f}   judge-human bias {bias:+.2f}")
    if bias > 0.3:
        print("  judge is more lenient than you - report its scores as an upper bound")
    elif bias < -0.3:
        print("  judge is harsher than you")

    by_tier: dict[str, list] = defaultdict(list)
    for h, j in pairs:
        by_tier[h["tier"]].append((h["overall"], j["overall"]))
    print("\nper tier (human mean / judge mean):")
    for t, xs in sorted(by_tier.items()):
        hm = sum(x for x, _ in xs) / len(xs)
        jm = sum(y for _, y in xs) / len(xs)
        print(f"  {t:<20}{hm:.2f} / {jm:.2f}   n={len(xs)}")
        out.setdefault("per_tier", {})[t] = {"human": hm, "judge": jm, "n": len(xs)}
    return out


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties averaged. Small enough to not need scipy."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


if __name__ == "__main__":
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="score a blind subset yourself")
    ap.add_argument("--n", type=int, default=40, help="replies to score (spread over tiers)")
    ap.add_argument("--agreement", action="store_true", help="judge vs human")
    args = ap.parse_args()
    if args.human:
        human(args.n)
    elif args.agreement:
        agreement()
    else:
        ap.print_help()
