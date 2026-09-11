# -*- coding: utf-8 -*-
"""LLM first-pass labelling of the golden set, against ANNOTATION_GUIDE.md.

This is NOT the hand-labelled golden set the assignment asks for. It is round
1 of a two-round protocol in which the human round is round 2:

    python label_golden.py                  # LLM labels -> data/golden_r1.jsonl
    python annotate.py --round 2 --blind    # human labels the same 250, blind
    python agreement.py                     # human-vs-LLM kappa
    python evaluate.py --gold data/golden_r2.jsonl   # score against the HUMAN labels

Why do it this way rather than the human labelling first: the human pass is
the expensive one, and doing it second, blind, turns it into both the ground
truth and a measurement of how far the LLM's labels can be trusted. Every
row written here carries `annotator: <model>` so provenance is never in
doubt, and the report says so.

The prompt is the annotation guide verbatim, so human and model are held to
the same definitions and the same tie-break rules.

Caveat for the report: if the same model also serves as the intent
classifier under test, intent accuracy against round 1 is circular. Score
against round 2, or use a different model for one of the two roles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from annotate import INTENTS, load_pool, out_path
from llm import BACKENDS, Client, utf8_console

HERE = Path(__file__).parent
GUIDE = HERE / "ANNOTATION_GUIDE.md"

SYSTEM_PREFIX = (
    "You are the annotator for a golden evaluation set. Follow the guide "
    "below exactly, including its tie-break rules. Label what a competent "
    "support organisation SHOULD do, not what an automated system would do.\n\n"
    "=== ANNOTATION GUIDE ===\n"
)


def build_prompt(batch: list[dict]) -> str:
    lines = ["MESSAGES:"]
    for i, r in enumerate(batch, 1):
        lines.append(f"{i}. [{r['language']}] {r['text']}")
    lines.append(
        f"\nReturn ONLY a JSON array with exactly {len(batch)} objects, in order. "
        "Keys: n (message number), intent (one of: " + ", ".join(INTENTS) + "), "
        "action (auto_handle or escalate), multi (true if more than one distinct "
        "problem), unsure (true if you would not bet on the label), "
        "rationale (under 12 words)."
    )
    return "\n".join(lines)


def parse(raw: str, batch: list[dict]) -> list[dict]:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("["), s.rfind("]")
    if a == -1 or b == -1:
        raise ValueError(f"no JSON array: {raw[:200]}")
    out = []
    for obj in json.loads(s[a:b + 1]):
        n = int(obj.get("n", 0))
        if not 1 <= n <= len(batch):
            continue
        intent = str(obj.get("intent", "")).strip()
        action = str(obj.get("action", "")).strip()
        if intent not in INTENTS or action not in ("auto_handle", "escalate"):
            continue
        out.append((n, intent, action, bool(obj.get("multi")), bool(obj.get("unsure")),
                    str(obj.get("rationale", ""))[:120]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--backend", choices=list(BACKENDS), default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = sorted(load_pool(), key=lambda r: r["order"])
    path = out_path(1)
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("annotator") is None:
                    raise SystemExit(
                        f"{path} already contains human labels; refusing to mix. "
                        "Delete it or label round 2 instead.")
                done.add(r["id"])
    todo = [r for r in pool if r["id"] not in done]
    print(f"golden pool {len(pool)}  labelled {len(done)}  to label {len(todo)}")
    if not todo:
        return 0

    system = SYSTEM_PREFIX + GUIDE.read_text(encoding="utf-8")
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    if args.dry_run:
        print("\n--- SYSTEM ---\n" + system[:1500] + "\n...\n--- USER ---\n" + build_prompt(batches[0]))
        return 0

    client = Client(backend=args.backend, model=args.model)
    print(f"backend={client.backend} model={client.model} batches={len(batches)}")
    t0, failed = time.time(), 0
    with path.open("a", encoding="utf-8") as sink:
        for bi, batch in enumerate(batches, 1):
            rows = []
            for attempt in (1, 2, 3):
                try:
                    rows = parse(client.complete(system, build_prompt(batch), max_tokens=4096), batch)
                    break
                except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
                    if attempt == 3:
                        print(f"  batch {bi} FAILED: {e}", file=sys.stderr)
            got = {n for n, *_ in rows}
            # Stragglers one at a time, as in judge.py.
            for k in range(1, len(batch) + 1):
                if k not in got:
                    try:
                        one = parse(client.complete(system, build_prompt([batch[k - 1]]), max_tokens=512),
                                    [batch[k - 1]])
                        rows += [(k, *rest) for _n, *rest in one]
                    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError):
                        failed += 1
            for n, intent, action, multi, unsure, why in rows:
                r = batch[n - 1]
                sink.write(json.dumps({
                    "id": r["id"], "text": r["text"], "stratum": r["stratum"],
                    "weight": r["weight"], "language": r["language"],
                    "intent": intent, "action": action, "multi": multi, "unsure": unsure,
                    "round": 1, "annotator": client.model, "llm_rationale": why,
                }, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"  batch {bi}/{len(batches)}  +{len(rows)}  ({client.usage})")
    print(f"\ndone in {time.time() - t0:.0f}s  failed {failed}  -> {path}")
    print("Next: python annotate.py --round 2 --blind   (your pass, blind)")
    return 0


if __name__ == "__main__":
    utf8_console()
    raise SystemExit(main())
