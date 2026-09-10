# -*- coding: utf-8 -*-
"""Keyboard-driven annotation tool for the golden evaluation set.

Two labels per message, because the system makes two decisions and both need
ground truth:

    intent  - one of the 10 taxonomy labels
    action  - auto_handle or escalate (the policy has no ground truth without this)

Plus two flags that matter for honest reporting:

    multi   - the message contains more than one distinct problem
    unsure  - you would not bet on this label

`multi` and `unsure` are not noise to be cleaned up later. They quantify the
ceiling on achievable accuracy, and a golden set that hides them will flatter
every model scored against it.

Round 2 (--round 2 --blind) re-presents the same messages in a different order
with your round-1 answers hidden, so agreement.py can compute self-agreement.
A solo annotator cannot report inter-annotator agreement, but intra-annotator
agreement is measurable and is the honest substitute.

Usage:
    python annotate.py                      # round 1, resumes where you left off
    python annotate.py --round 2 --blind    # after a day
    python annotate.py --review             # revisit only rows flagged unsure
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llm import utf8_console

HERE = Path(__file__).parent
POOL = HERE / "data" / "golden_pool.jsonl"

INTENTS = [
    "delivery_delayed",
    "delivery_not_received",
    "item_damaged_wrong_missing",
    "order_cancel_or_change",
    "refund_or_return",
    "payment_or_charge",
    "prime_membership",
    "account_access",
    "service_complaint_or_feedback",
    "other",
]
HINTS = {
    "delivery_delayed": "still waiting, late or will miss promised date",
    "delivery_not_received": "tracking says DELIVERED but customer has nothing",
    "item_damaged_wrong_missing": "arrived but damaged / wrong / incomplete",
    "order_cancel_or_change": "cancel or modify pre-delivery, or Amazon cancelled it",
    "refund_or_return": "money owed back, or how do I send it back",
    "payment_or_charge": "card declined, double charge, gift card (NOT Prime fees)",
    "prime_membership": "Prime signup/charge/cancel/benefits, Prime Video",
    "account_access": "locked out, hacked, password, verification",
    "service_complaint_or_feedback": "general rant or praise, no single actionable case",
    "other": "unintelligible, fragment, or not a support request",
}


def out_path(rnd: int) -> Path:
    return HERE / "data" / f"golden_r{rnd}.jsonl"


def load_pool() -> list[dict]:
    if not POOL.exists():
        raise SystemExit(f"missing {POOL} - run sample_golden.py first")
    return [json.loads(l) for l in POOL.open(encoding="utf-8") if l.strip()]


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def menu() -> str:
    lines = ["", "  INTENT" + " " * 26 + "(number to pick)"]
    for i, name in enumerate(INTENTS, 1):
        lines.append(f"   {i:>2}. {name:<32}{HINTS[name]}")
    lines.append("   s. skip   q. save and quit")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    ap.add_argument("--blind", action="store_true",
                    help="round 2: do not show round-1 answers")
    ap.add_argument("--review", action="store_true",
                    help="revisit only rows flagged unsure")
    args = ap.parse_args()

    pool = load_pool()
    path = out_path(args.round)
    done = load_done(path)

    if args.round == 2:
        # Different order in round 2 so position cannot cue recall.
        random.seed(99)
        random.shuffle(pool)
        r1 = load_done(out_path(1))
        if not r1:
            raise SystemExit("round 1 not started - run `python annotate.py` first")
        pool = [p for p in pool if p["id"] in r1]

    if args.review:
        todo = [p for p in pool if done.get(p["id"], {}).get("unsure")]
        done = {k: v for k, v in done.items() if not v.get("unsure")}
    else:
        todo = [p for p in pool if p["id"] not in done]

    total = len(pool)
    print(f"\nround {args.round}  |  {len(done)}/{total} labelled  |  {len(todo)} to go")
    print("Type the intent number, then a/e for auto/escalate.")
    print("Suffix flags: 'm' multi-intent, 'u' unsure.  e.g.  '1 e mu'")

    with path.open("a", encoding="utf-8") as sink:
        for n, row in enumerate(todo, 1):
            print("\n" + "=" * 78)
            print(f"[{n}/{len(todo)}]  id={row['id']}  stratum={row['stratum']}"
                  f"  lang={row['language']}")
            print("=" * 78)
            print(f"\n  {row['text']}\n")
            if args.round == 2 and not args.blind:
                prev = load_done(out_path(1)).get(row["id"], {})
                print(f"  (round 1 said: {prev.get('intent')} / {prev.get('action')})")
            print(menu())

            while True:
                raw = input("\n> ").strip().lower()
                if raw == "q":
                    print(f"\nsaved to {path}")
                    return
                if raw == "s":
                    break
                parts = raw.split()
                if not parts or not parts[0].isdigit():
                    print("  need an intent number, e.g. '3 e' or '3 a u'")
                    continue
                k = int(parts[0])
                if not 1 <= k <= len(INTENTS):
                    print(f"  intent must be 1-{len(INTENTS)}")
                    continue
                action = None
                for p in parts[1:]:
                    if p in ("a", "auto", "auto_handle"):
                        action = "auto_handle"
                    elif p in ("e", "esc", "escalate"):
                        action = "escalate"
                if action is None:
                    print("  need a/e for auto_handle or escalate")
                    continue
                flags = "".join(parts[1:])
                rec = {
                    "id": row["id"],
                    "text": row["text"],
                    "stratum": row["stratum"],
                    "weight": row["weight"],
                    "language": row["language"],
                    "intent": INTENTS[k - 1],
                    "action": action,
                    "multi": "m" in flags,
                    "unsure": "u" in flags,
                    "round": args.round,
                }
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                extra = []
                if rec["multi"]:
                    extra.append("multi")
                if rec["unsure"]:
                    extra.append("unsure")
                print(f"  -> {rec['intent']} / {action}"
                      + (f"  [{', '.join(extra)}]" if extra else ""))
                break

    print(f"\nround {args.round} complete -> {path}")
    if args.round == 1:
        print("Wait a day, then: python annotate.py --round 2 --blind")
    else:
        print("Now: python agreement.py")


if __name__ == "__main__":
    utf8_console()
    main()
