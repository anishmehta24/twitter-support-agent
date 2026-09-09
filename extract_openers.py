# -*- coding: utf-8 -*-
"""Extract AmazonHelp conversation openers for taxonomy work.

An "opener" is an inbound customer tweet that starts a thread
(in_response_to_tweet_id empty) and addresses @AmazonHelp.

Writes a compact TSV so downstream steps never touch the 516 MB source again.
"""
import csv
import html
import re
import sys
from pathlib import Path

SRC = Path(r"C:\Users\BIT\oss\hiver\data\twcs.csv")
OUT = Path(r"C:\Users\BIT\oss\hiver\data\amazon_openers.tsv")
BRAND = "AmazonHelp"
csv.field_size_limit(10_000_000)

MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Strip handles/URLs so clustering keys on content, not addressing."""
    t = html.unescape(text)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = WS.sub(" ", t)
    return t.strip()


rows = kept = 0
seen = set()
with SRC.open("r", encoding="utf-8", errors="replace", newline="") as fh, \
     OUT.open("w", encoding="utf-8", newline="") as out:
    w = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    w.writerow(["tweet_id", "author_id", "created_at", "raw", "clean"])
    for row in csv.DictReader(fh):
        rows += 1
        if rows % 500_000 == 0:
            print(f"  ...{rows:,}", file=sys.stderr, flush=True)

        if (row.get("inbound") or "").strip().lower() != "true":
            continue
        if (row.get("in_response_to_tweet_id") or "").strip():
            continue

        raw = row.get("text") or ""
        m = MENTION.search(raw)
        if not m or m.group(0)[1:] != BRAND:
            continue

        c = clean(raw)
        if len(c) < 15:            # drop "help!" style stubs
            continue
        key = c.lower()[:120]
        if key in seen:            # near-duplicate spam
            continue
        seen.add(key)

        kept += 1
        w.writerow([row.get("tweet_id", ""), row.get("author_id", ""),
                    row.get("created_at", ""), raw.replace("\t", " "), c])

print(f"\nscanned {rows:,} rows -> kept {kept:,} unique openers")
print(f"written: {OUT}")
