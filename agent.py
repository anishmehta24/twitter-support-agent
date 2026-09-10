# -*- coding: utf-8 -*-
"""The support agent, end to end: classify -> retrieve -> decide -> draft.

Everything else in the repo is a component. This is the thing the assignment
asked for, and the thing evaluate.py scores.

Order matters. Retrieval runs before the decision because the decision needs
the grounding score (gate 4). The reply is drafted even when the decision is
"escalate", so a human reviewer sees a suggested draft and the judge can score
it - but the `action` field is authoritative: an escalated draft is never sent.

    python agent.py "my parcel says delivered but i never got it"
    python agent.py --classifier keyword --tier nearest_neighbour "..."
"""
from __future__ import annotations

import argparse
import json

from classify import Classifier, make
from escalation import decide
from llm import utf8_console
from reply import ReplyGenerator
from retrieve import Retriever


class Agent:
    def __init__(self, classifier: Classifier | str = "llm",
                 tier: str = "grounded_llm", k: int = 5,
                 retriever: Retriever | None = None, client=None):
        self.r = retriever or Retriever.load()
        self.clf = make(classifier, client=client) if isinstance(classifier, str) else classifier
        self.gen = ReplyGenerator(self.r, tier=tier, k=k, client=client)
        self.k = k

    def handle_batch(self, texts: list[str], ids: list[str] | None = None,
                     draft: bool = True) -> list[dict]:
        """Batched so the LLM classifier sends 10 messages per call."""
        if hasattr(self.clf, "predict") and self.clf.name == "llm":
            preds = self.clf.predict(texts, ids=ids)
        else:
            preds = self.clf.predict(texts)
        out = []
        for text, p in zip(texts, preds):
            precedents = self.r.search(text, k=self.k)
            grounding = precedents[0].score if precedents else 0.0
            d = decide(p.intent, p.confidence, text, grounding, language=p.language)
            rec = {
                "text": text,
                "intent": p.intent,
                "confidence": p.confidence,
                "language": p.language,
                "grounding": grounding,
                "action": d.action.value,
                "gate": d.gate,
                "reason": d.reason,
                "triggers": d.triggers,
            }
            if draft:
                rec["reply"] = self.gen.generate(text, precedents=precedents)["reply"]
                rec["precedents"] = [{"customer": q.customer, "reply": q.reply,
                                      "score": q.score} for q in precedents[:3]]
            out.append(rec)
        return out

    def handle(self, text: str, draft: bool = True) -> dict:
        return self.handle_batch([text], draft=draft)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="llm", choices=["majority", "keyword", "llm"])
    ap.add_argument("--tier", default="grounded_llm",
                    choices=["trivial", "nearest_neighbour", "grounded_llm"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("message", nargs="+")
    args = ap.parse_args()

    agent = Agent(classifier=args.classifier, tier=args.tier)
    out = agent.handle(" ".join(args.message))
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    print(f"\nintent     : {out['intent']}  (conf {out['confidence']:.2f}, lang {out['language']})")
    print(f"grounding  : {out['grounding']:.3f}")
    print(f"action     : {out['action'].upper()}  [{out['gate']}]")
    print(f"reason     : {out['reason']}")
    print(f"\ndraft ({args.tier}):\n  {out['reply']}")
    if out["action"] == "escalate":
        print("\n  ^ draft shown for the human reviewer; NOT sent automatically")


if __name__ == "__main__":
    utf8_console()
    main()
