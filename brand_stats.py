# -*- coding: utf-8 -*-
"""Streaming brand statistics for the TWCS (Customer Support on Twitter) dataset.

Memory-light on purpose: single pass, only counters held in RAM. No pandas.

Schema: tweet_id, author_id, inbound, created_at, text, response_tweet_id,
        in_response_to_tweet_id

Brands have a handle as author_id ("AppleSupport"); customers have numeric ids.
Inbound customer messages are attributed to a brand via the first @mention.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CSV = Path(r"C:\Users\BIT\oss\hiver\data\twcs.csv")
csv.field_size_limit(10_000_000)

MENTION = re.compile(r"@(\w+)")

outbound = Counter()          # brand -> replies sent
outbound_isreply = Counter()  # brand -> replies that answer another tweet
inbound_to = Counter()        # brand -> customer messages addressed to it
inbound_openers = Counter()   # brand -> customer messages starting a thread
customers = defaultdict(set)  # brand -> distinct customer ids (capped)
CUST_CAP = 60_000             # stop growing a brand's set past this

rows = 0
with CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows += 1
        if rows % 500_000 == 0:
            print(f"  ...{rows:,} rows", file=sys.stderr, flush=True)

        author = (row.get("author_id") or "").strip()
        is_inbound = (row.get("inbound") or "").strip().lower() == "true"
        in_reply_to = (row.get("in_response_to_tweet_id") or "").strip()

        if not is_inbound:
            # Brand-authored reply
            if author and not author.isdigit():
                outbound[author] += 1
                if in_reply_to:
                    outbound_isreply[author] += 1
        else:
            # Customer message: attribute by first @mention of a handle
            text = row.get("text") or ""
            m = MENTION.search(text)
            if m:
                handle = m.group(1)
                inbound_to[handle] += 1
                if not in_reply_to:
                    inbound_openers[handle] += 1
                if len(customers[handle]) < CUST_CAP and author.isdigit():
                    customers[handle].add(author)

print(f"\ntotal rows: {rows:,}\n")

# Real brands = handles that actually authored outbound replies
brands = [b for b, n in outbound.items() if n >= 500]
brands.sort(key=lambda b: outbound[b], reverse=True)

hdr = (f"{'BRAND':<22}{'REPLIES':>10}{'INBOUND':>10}{'OPENERS':>9}"
       f"{'CUSTOMERS':>11}{'REPLY/OPENER':>14}{'THREADED%':>11}")
print(hdr)
print("-" * len(hdr))
for b in brands[:30]:
    rep = outbound[b]
    inb = inbound_to.get(b, 0)
    opn = inbound_openers.get(b, 0)
    cst = len(customers.get(b, ()))
    ratio = (rep / opn) if opn else 0.0
    threaded = (100.0 * outbound_isreply[b] / rep) if rep else 0.0
    cap = "+" if cst >= CUST_CAP else " "
    print(f"{b:<22}{rep:>10,}{inb:>10,}{opn:>9,}{cst:>10,}{cap}{ratio:>14.2f}{threaded:>10.1f}%")
