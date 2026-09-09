# -*- coding: utf-8 -*-
"""Build (customer message -> AmazonHelp's actual reply) pairs.

These pairs are the grounding corpus. "Draft a reply grounded in how the brand
has historically resolved similar issues" means: find past messages like this
one, and look at what Amazon actually said.

Two passes so memory stays bounded. Pass 1 keeps only AmazonHelp's outbound
tweets keyed by id (~170k short strings). Pass 2 streams inbound tweets and
joins on response_tweet_id, which can list several replies.

An important caveat for the report: the brand's historical reply is what Amazon
*did*, not what was *right*. Some of these are bad answers. Retrieval-grounded
generation inherits that, and no metric computed against these replies can see
it.
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "data" / "twcs.csv"
OUT = HERE / "data" / "amazon_pairs.jsonl"
BRAND = "AmazonHelp"
csv.field_size_limit(10_000_000)

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\s+")
DM = re.compile(r"\b(dm|direct message|private message|send us a (dm|message))\b", re.I)

# Mid-thread pleasantries ("Ok, thanks for your help...") pair with equally
# contentless replies ("Not a problem!"). They are ~real conversation but carry
# no resolution, so retrieving against them is pure noise.
PLEASANTRY = re.compile(
    r"^(ok(ay)?|thanks?|thank you|thx|ty|cheers|great|cool|perfect|"
    r"got it|will do|done|yes|no|yep|nope|sure|awesome|nice|lol|"
    r"appreciate it|much appreciated)\b", re.I)
PROBLEM = re.compile(
    r"\b(order|deliver\w*|refund|return|charg\w*|account|package|parcel|"
    r"item|cancel\w*|prime|payment|money|late|missing|damaged|broken|"
    r"wrong|help|issue|problem|error|why|when|how)\b", re.I)


def is_pleasantry(t: str) -> bool:
    """Short, opens with an acknowledgement, and names no problem."""
    return bool(PLEASANTRY.match(t)) and len(t) < 90 and not PROBLEM.search(t)


def clean(t: str) -> str:
    t = html.unescape(t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    return WS.sub(" ", t).strip()


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} - run scripts/fetch_data.sh")

    # ---- pass 1: AmazonHelp outbound replies, id -> text -------------------
    replies: dict[str, str] = {}
    rows = 0
    with SRC.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            if rows % 500_000 == 0:
                print(f"  pass1 ...{rows:,}", file=sys.stderr, flush=True)
            if (row.get("author_id") or "").strip() != BRAND:
                continue
            if (row.get("inbound") or "").strip().lower() == "true":
                continue
            tid = (row.get("tweet_id") or "").strip()
            if tid:
                replies[tid] = row.get("text") or ""
    print(f"pass 1: {len(replies):,} AmazonHelp replies indexed")

    # ---- pass 2: inbound -> reply join ------------------------------------
    kept = skipped_dm = skipped_short = skipped_pleasantry = 0
    openers = 0
    rows = 0
    with SRC.open("r", encoding="utf-8", errors="replace", newline="") as fh, \
         OUT.open("w", encoding="utf-8") as out:
        for row in csv.DictReader(fh):
            rows += 1
            if rows % 500_000 == 0:
                print(f"  pass2 ...{rows:,}", file=sys.stderr, flush=True)
            if (row.get("inbound") or "").strip().lower() != "true":
                continue
            resp = (row.get("response_tweet_id") or "").strip()
            if not resp:
                continue

            # A customer tweet can have several brand replies listed.
            targets = [r for r in (x.strip() for x in resp.split(",")) if r in replies]
            if not targets:
                continue

            cust_raw = row.get("text") or ""
            if MENTION.search(cust_raw) is None:
                continue
            if MENTION.search(cust_raw).group(0)[1:] != BRAND:
                continue

            cust = clean(cust_raw)
            if len(cust) < 15:
                skipped_short += 1
                continue
            if is_pleasantry(cust):
                skipped_pleasantry += 1
                continue

            reply_raw = " ".join(replies[t] for t in targets)
            reply = clean(reply_raw)
            if len(reply) < 15:
                skipped_short += 1
                continue
            # A pure DM deflection teaches the generator nothing useful.
            if DM.search(reply) and len(reply) < 160:
                skipped_dm += 1
                continue

            # Openers are self-contained problem statements. Mid-thread turns
            # assume context the retriever will not have at query time, so the
            # index prefers openers and falls back to the rest.
            is_opener = not (row.get("in_response_to_tweet_id") or "").strip()
            openers += int(is_opener)

            out.write(json.dumps({
                "id": row.get("tweet_id", ""),
                "customer": cust,
                "reply": reply,
                "is_opener": is_opener,
                "n_reply_tweets": len(targets),
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\npass 2: {kept:,} pairs written")
    print(f"  of which openers        : {openers:,} ({100*openers/max(kept,1):.1f}%)")
    print(f"  skipped (DM deflection) : {skipped_dm:,}")
    print(f"  skipped (too short)     : {skipped_short:,}")
    print(f"  skipped (pleasantry)    : {skipped_pleasantry:,}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
