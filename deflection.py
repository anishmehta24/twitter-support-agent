# -*- coding: utf-8 -*-
"""How substantive are each brand's replies?

A reply that says "please DM us" is useless as grounding material for a reply
generator: the actual resolution happened in a private channel the dataset
never sees. This measures, per candidate brand:

  DEFLECT%  - replies pushing the user to DM/private channel
  LINK%     - replies whose substance is a URL (help-page deflection)
  APOLOGY%  - replies that are pure apology with no action
  MEDIAN_CH - median reply length in characters
  SUBSTANT% - replies that are none of the above (the usable pool)
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

CSV = Path(r"C:\Users\BIT\oss\hiver\data\twcs.csv")
csv.field_size_limit(10_000_000)

BRANDS = {
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta",
    "Tesco", "AmericanAir", "TMobileHelp", "comcastcares", "British_Airways",
    "SouthwestAir", "Ask_Spectrum", "XboxSupport", "hulu_support",
    "AskPlayStation", "VirginTrains",
}

DM = re.compile(r"\b(dm|direct message|private message|pm us|send us a (dm|message)|"
                r"follow(ed)? (us )?(and|so we can) dm|via dm|in a dm|dm'?d?|"
                r"reach out .{0,20}(privately|via dm))\b", re.I)
LINK = re.compile(r"https?://\S+")
APOLOGY = re.compile(r"\b(sorry|apolog|regret|unfortunate)\w*\b", re.I)
ACTION = re.compile(r"\b(refund|credit|rebook|resend|replac|reset|cancel|track|"
                    r"deliver|arriv|schedul|confirm|process|issued|updat|fix|"
                    r"restor|activat|charg|order|number|status)\w*\b", re.I)

deflect = Counter(); link = Counter(); apology = Counter()
total = Counter(); substant = Counter()
lengths = defaultdict(list)
LEN_CAP = 40_000

rows = 0
with CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
    for row in csv.DictReader(fh):
        rows += 1
        if rows % 500_000 == 0:
            print(f"  ...{rows:,}", file=sys.stderr, flush=True)

        author = (row.get("author_id") or "").strip()
        if author not in BRANDS:
            continue
        if (row.get("inbound") or "").strip().lower() == "true":
            continue

        text = (row.get("text") or "").strip()
        total[author] += 1
        if len(lengths[author]) < LEN_CAP:
            lengths[author].append(len(text))

        stripped = LINK.sub("", text).strip()
        is_dm = bool(DM.search(text))
        is_link = bool(LINK.search(text)) and len(stripped) < 40
        is_apology = bool(APOLOGY.search(text)) and not ACTION.search(text) and len(text) < 120

        if is_dm:
            deflect[author] += 1
        if is_link:
            link[author] += 1
        if is_apology:
            apology[author] += 1
        if not (is_dm or is_link or is_apology):
            substant[author] += 1

print(f"\nscanned {rows:,} rows\n")
hdr = (f"{'BRAND':<20}{'REPLIES':>9}{'DEFLECT%':>10}{'LINK%':>8}"
       f"{'APOLOGY%':>10}{'MEDIAN_CH':>11}{'SUBSTANT%':>11}")
print(hdr); print("-" * len(hdr))
for b in sorted(total, key=lambda x: substant[x] / total[x] if total[x] else 0, reverse=True):
    n = total[b]
    if not n:
        continue
    print(f"{b:<20}{n:>9,}{100*deflect[b]/n:>9.1f}%{100*link[b]/n:>7.1f}%"
          f"{100*apology[b]/n:>9.1f}%{median(lengths[b]):>11.0f}{100*substant[b]/n:>10.1f}%")
