# -*- coding: utf-8 -*-
"""Reply generation at three tiers, so the system has something to beat.

The assignment requires at least two baselines, a trivial one and a simple one.
Building them first is not box-ticking: until the simple baseline is measured,
the LLM's score has no meaning.

  tier 1  trivial            single canned reply for everything
  tier 2  nearest_neighbour  return the brand's actual historical reply verbatim
  tier 3  grounded_llm       compose from the top-k retrieved precedents

Expect tier 2 to be uncomfortably strong. Support corpora are repetitive, and
returning what Amazon genuinely said to a near-identical message is a hard
bar. If tier 3 cannot beat it, that is a real finding and belongs in the
report, not something to hide behind a friendlier metric.

A caveat that no automated metric here can see: the retrieved "resolutions" are
frequently soft redirects ("please reach out here"). A generator grounded in
them learns to deflect politely, and will score well against a judge that
rewards tone and grounding while resolving nothing.
"""
from __future__ import annotations

import argparse
import re

from llm import utf8_console
from retrieve import Retriever

SIGNOFF = re.compile(r"\s*\^[A-Z]{2,3}\b")

TRIVIAL_REPLY = (
    "Sorry for the trouble! Please reach out to us here so we can help: "
    "https://amazon.com/contact-us"
)

SYSTEM = (
    "You draft short public replies for Amazon's Twitter support account.\n\n"
    "Rules:\n"
    "- Ground the reply in the provided precedents: mirror how Amazon actually "
    "handled similar cases, including tone and the next step offered.\n"
    "- Never invent order numbers, refund amounts, dates, or policies that do "
    "not appear in the precedents.\n"
    "- Never ask the customer to post personal details publicly.\n"
    "- Never address the customer by a name unless it appears in the NEW "
    "MESSAGE itself; names in precedents belong to other customers.\n"
    "- Do not copy agent sign-offs like '^AB' from the precedents.\n"
    "- Under 280 characters. No hashtags. No emoji unless the precedents use them.\n"
    "- If the precedents do not cover the situation, say plainly that a human "
    "will pick it up rather than guessing."
)


def build_prompt(message: str, precedents) -> str:
    lines = ["PRECEDENTS - how Amazon handled similar messages:\n"]
    for i, p in enumerate(precedents, 1):
        lines.append(f"[{i}] (similarity {p.score:.2f})")
        lines.append(f"  customer: {p.customer}")
        lines.append(f"  amazon:   {p.reply}\n")
    lines.append(f"NEW MESSAGE:\n  {message}\n")
    lines.append("Draft Amazon's reply. Output the reply text only.")
    return "\n".join(lines)


class ReplyGenerator:
    def __init__(self, retriever: Retriever, tier: str = "grounded_llm",
                 k: int = 5, client=None):
        self.r = retriever
        self.tier = tier
        self.k = k
        self._client = client

    @property
    def client(self):
        # Imported lazily so tiers 1 and 2 need no LLM backend at all.
        if self._client is None:
            from llm import Client
            self._client = Client()
        return self._client

    def generate(self, message: str, precedents=None) -> dict:
        # Caller may pass precedents it already retrieved (agent.py does, so
        # the decision and the draft are grounded in the same evidence).
        if precedents is None:
            precedents = self.r.search(message, k=self.k)
        grounding = precedents[0].score if precedents else 0.0

        if self.tier == "trivial":
            text = TRIVIAL_REPLY
        elif self.tier == "nearest_neighbour":
            text = precedents[0].reply if precedents else TRIVIAL_REPLY
        elif self.tier == "grounded_llm":
            text = self.client.complete(
                SYSTEM, build_prompt(message, precedents), max_tokens=300
            ).strip()
            # Small models copy the precedents' agent initials ("^DW") despite
            # the rule above. Strip them: a generated reply has no agent.
            text = SIGNOFF.sub("", text).strip()
        else:
            raise ValueError(f"unknown tier: {self.tier}")

        return {
            "tier": self.tier,
            "reply": text,
            "grounding": grounding,
            "precedents": [{"customer": p.customer, "reply": p.reply,
                            "score": p.score} for p in precedents],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all",
                    choices=["trivial", "nearest_neighbour", "grounded_llm", "all"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("message", nargs="*")
    args = ap.parse_args()

    r = Retriever.load()
    messages = [" ".join(args.message)] if args.message else [
        "my parcel says delivered but i never got it",
        "charged twice for prime membership this month",
        "asdkjh random gibberish nothing to do with anything",
    ]
    tiers = (["trivial", "nearest_neighbour", "grounded_llm"]
             if args.tier == "all" else [args.tier])

    for m in messages:
        print("\n" + "=" * 78)
        print(f"MESSAGE: {m}")
        print("=" * 78)
        for t in tiers:
            try:
                out = ReplyGenerator(r, tier=t, k=args.k).generate(m)
            except SystemExit as e:            # no LLM backend configured
                print(f"\n  [{t}] skipped: {e}")
                continue
            print(f"\n  [{t}]  grounding={out['grounding']:.3f}")
            print(f"    {out['reply'][:260]}")


if __name__ == "__main__":
    utf8_console()
    main()
