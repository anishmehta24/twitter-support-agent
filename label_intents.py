# -*- coding: utf-8 -*-
"""LLM labelling of AmazonHelp openers against the candidate taxonomy.

Purpose: validate the taxonomy derived from clustering. If `other` comes back
above ~15%, the taxonomy is missing a category and needs another pass.

Design notes (these are the interesting decisions, worth keeping for the log):

  * Stratified sampling by cluster, so rare intents are represented. A uniform
    random sample is dominated by the 31% residual cluster and tells you little.
  * Batched prompts (default 10 messages per call). Cuts cost ~5x versus one
    call per message, because the taxonomy definition dominates the prompt and
    is sent once per batch instead of once per message.
  * Resumable. Every result is appended to JSONL immediately; a re-run skips
    ids already present. An interrupted run never costs twice.
  * Token accounting per run, so cost per labelled example is measured rather
    than assumed.
  * The model returns a confidence and a short rationale. Low-confidence rows
    are exactly the ones worth spending human labelling time on.

Backends, auto-detected in order: ANTHROPIC_API_KEY, OPENAI_API_KEY, local
Ollama on 127.0.0.1:11434. Override with --backend.

Usage:
    python label_intents.py --n 500                 # label 500, stratified
    python label_intents.py --n 500 --dry-run       # print one prompt, no calls
    python label_intents.py --n 2000 --batch 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

from llm import BACKENDS, Client

HERE = Path(__file__).parent
CLUSTERS = HERE / "data" / "amazon_clusters.tsv"
OUT = HERE / "data" / "labels.jsonl"
csv.field_size_limit(10_000_000)

# --------------------------------------------------------------------------
# Taxonomy. Descriptions matter more than names - they are the actual spec the
# model is labelling against, and they are what a human annotator must follow
# too if the golden set is to be comparable.
# --------------------------------------------------------------------------
TAXONOMY = {
    "delivery_delayed": (
        "Order has not arrived yet and is late, or is predicted to miss a "
        "promised/guaranteed date. The customer is still waiting."
    ),
    "delivery_not_received": (
        "Tracking says DELIVERED (or the driver claims delivery) but the "
        "customer does not have the item. Includes stolen/misdelivered parcels."
    ),
    "item_damaged_wrong_missing": (
        "Item physically arrived but is damaged, defective, the wrong item, or "
        "part of the order is missing from the box."
    ),
    "order_cancel_or_change": (
        "Wants to cancel, modify, or query an order's status pre-delivery; or "
        "an order was cancelled unexpectedly by Amazon."
    ),
    "refund_or_return": (
        "Refund not received, refund delayed, or asking how to return an item. "
        "Money owed back to the customer, or getting goods back to Amazon."
    ),
    "payment_or_charge": (
        "Payment declined, card/bank issue, unexpected or duplicate charge, "
        "gift card or Amazon Pay balance problems. NOT Prime fees."
    ),
    "prime_membership": (
        "Anything about Prime specifically: signup, unwanted Prime charge, "
        "cancellation, benefits, or Prime Video content and playback."
    ),
    "account_access": (
        "Cannot log in, account locked, suspended, hacked, password or "
        "verification-code problems, account closure requests."
    ),
    "service_complaint_or_feedback": (
        "General praise or a general rant about Amazon/customer service with no "
        "single specific actionable case attached. Use only when no other "
        "intent clearly fits the substance."
    ),
    "other": (
        "Genuinely does not fit any category above, is unintelligible, is a "
        "fragment lacking context, or is not a support request at all."
    ),
}

SYSTEM = (
    "You are labelling customer-support tweets sent to Amazon's support account "
    "for a dataset annotation task. Assign exactly one intent per message.\n\n"
    "Rules:\n"
    "- Choose the intent matching the customer's PRIMARY problem, not incidental mentions.\n"
    "- 'delivery_delayed' means still waiting; 'delivery_not_received' means marked "
    "delivered but absent. This distinction matters - read carefully.\n"
    "- Prime fees are 'prime_membership', not 'payment_or_charge'.\n"
    "- Messages may be in any language. Label the intent regardless of language, "
    "and report the language.\n"
    "- Use 'other' rather than guessing when the message is a fragment or unclear.\n"
    "- confidence is your own 0.0-1.0 estimate that this label is correct."
)


def build_prompt(batch: list[tuple[str, str]]) -> str:
    lines = ["INTENT TAXONOMY:"]
    for name, desc in TAXONOMY.items():
        lines.append(f"- {name}: {desc}")
    lines.append("\nMESSAGES:")
    for i, (_id, text) in enumerate(batch, 1):
        lines.append(f"{i}. {text}")
    lines.append(
        "\nReturn ONLY a JSON array, one object per message, in order:\n"
        '[{"n":1,"intent":"<name>","language":"<iso639-1>","confidence":0.0,'
        '"rationale":"<max 12 words>"}]'
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
def parse_response(raw: str, batch: list[tuple[str, str]]) -> list[dict]:
    """Tolerant JSON extraction - models wrap arrays in prose or fences."""
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("["), s.rfind("]")
    if a == -1 or b == -1:
        raise ValueError(f"no JSON array in response: {raw[:200]}")
    items = json.loads(s[a:b + 1])

    out = []
    for obj in items:
        n = int(obj.get("n", 0))
        if not (1 <= n <= len(batch)):
            continue
        intent = str(obj.get("intent", "")).strip()
        if intent not in TAXONOMY:
            intent = "other"
        out.append({
            "id": batch[n - 1][0],
            "text": batch[n - 1][1],
            "intent": intent,
            "language": str(obj.get("language", "") or "")[:5],
            "confidence": float(obj.get("confidence", 0.0) or 0.0),
            "rationale": str(obj.get("rationale", ""))[:120],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="messages to label")
    ap.add_argument("--batch", type=int, default=10, help="messages per API call")
    ap.add_argument("--backend", choices=list(BACKENDS), default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    args = ap.parse_args()

    if not CLUSTERS.exists():
        raise SystemExit(f"missing {CLUSTERS} - run cluster_intents.py first")

    # ---- stratified sample by cluster --------------------------------------
    by_cluster: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with CLUSTERS.open("r", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            by_cluster[row["cluster"]].append((f"t{i}", row["text"]))

    random.seed(args.seed)
    per = max(1, args.n // len(by_cluster))
    sample: list[tuple[str, str]] = []
    for c, rows in sorted(by_cluster.items()):
        sample.extend(random.sample(rows, min(per, len(rows))))

    # Integer division drops a remainder (n=40 over 15 clusters gives 30), and
    # small clusters cannot fill their quota. Top up at random from whatever is
    # left so --n is honoured exactly.
    if len(sample) < args.n:
        chosen = {i for i, _ in sample}
        pool = [x for rows in by_cluster.values() for x in rows if x[0] not in chosen]
        sample.extend(random.sample(pool, min(args.n - len(sample), len(pool))))

    random.shuffle(sample)
    sample = sample[: args.n]

    done = set()
    if OUT.exists():
        with OUT.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [x for x in sample if x[0] not in done]
    print(f"sample={len(sample)}  already labelled={len(done)}  to label={len(todo)}")
    if not todo:
        return summarise()

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    if args.dry_run:
        print("\n--- SYSTEM ---\n" + SYSTEM)
        print("\n--- USER (first batch) ---\n" + build_prompt(batches[0]))
        print(f"\n({len(batches)} batches would be sent)")
        return 0

    client = Client(backend=args.backend, model=args.model)
    backend, model = client.backend, client.model

    print(f"backend={backend}  model={model}  batches={len(batches)}\n")

    tin = tout = failed = 0
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as sink:
        for bi, batch in enumerate(batches, 1):
            prompt = build_prompt(batch)
            for attempt in (1, 2, 3):
                try:
                    raw = client.complete(SYSTEM, prompt, max_tokens=4096, retries=1)
                    rows = parse_response(raw, batch)
                    tin = client.usage.input_tokens
                    tout = client.usage.output_tokens
                    for r in rows:
                        r["model"] = model
                        sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                    sink.flush()
                    print(f"  batch {bi}/{len(batches)}  +{len(rows)} labels "
                          f"({tin:,} in / {tout:,} out tokens)")
                    break
                except (RuntimeError, urllib.error.HTTPError,
                        urllib.error.URLError, ValueError, KeyError,
                        json.JSONDecodeError) as e:
                    if attempt == 3:
                        failed += len(batch)
                        print(f"  batch {bi} FAILED after 3 tries: {e}", file=sys.stderr)
                    else:
                        time.sleep(2 ** attempt)

    dt = time.time() - t0
    print(f"\ndone in {dt:.0f}s  tokens: {tin:,} in / {tout:,} out  failed: {failed}")
    if tin:
        n_lab = len(todo) - failed
        if n_lab:
            print(f"~{(tin + tout) / n_lab:.0f} tokens per labelled example")
    return summarise()


def summarise() -> int:
    if not OUT.exists():
        return 0
    rows = [json.loads(l) for l in OUT.open("r", encoding="utf-8") if l.strip()]
    if not rows:
        return 0
    print(f"\n=== {len(rows):,} labels in {OUT.name} ===")
    intents = Counter(r["intent"] for r in rows)
    for k, v in intents.most_common():
        conf = [r["confidence"] for r in rows if r["intent"] == k]
        avg = sum(conf) / len(conf) if conf else 0
        print(f"  {k:<32}{v:>6}  ({100*v/len(rows):>5.1f}%)  mean conf {avg:.2f}")

    other = 100 * intents.get("other", 0) / len(rows)
    print(f"\n  other rate: {other:.1f}%  "
          f"{'-> taxonomy likely missing a category' if other > 15 else '-> acceptable'}")

    langs = Counter(r.get("language", "") for r in rows)
    print("  languages: " + ", ".join(f"{k or '?'}={v}" for k, v in langs.most_common(6)))

    low = [r for r in rows if r["confidence"] < 0.6]
    print(f"  low-confidence (<0.6): {len(low)} "
          f"({100*len(low)/len(rows):.1f}%) <- prioritise these for human review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
