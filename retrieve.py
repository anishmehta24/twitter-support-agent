# -*- coding: utf-8 -*-
"""Retrieval over historical (customer message -> Amazon reply) pairs.

This is what "grounded in how the brand has historically resolved similar
issues" actually means: find past messages like this one and look at what
Amazon said.

It serves two purposes, and the second is easy to miss:

  1. Supplies precedents for the reply generator.
  2. Supplies the *grounding score* that escalation.py gate 4 depends on.
     A low top-1 similarity means there is no comparable precedent, which is
     exactly when the agent should not be replying unsupervised.

Index: TF-IDF -> TruncatedSVD -> L2 normalise, then exact cosine nearest
neighbours. Deliberately not embeddings: this runs on CPU in seconds, has no
API dependency, and keeps the 15-minute reproduction budget intact. Swapping in
sentence-transformers later is a one-function change, and comparing the two
would be a legitimate result.

Openers are indexed by default. Mid-thread turns assume context the retriever
does not have at query time ("yes it's the second one") and pollute results.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

HERE = Path(__file__).parent
PAIRS = HERE / "data" / "amazon_pairs.jsonl"
INDEX = HERE / "data" / "retrieval_index.pkl"
SEED = 42
N_COMPONENTS = 200


@dataclass
class Precedent:
    customer: str
    reply: str
    score: float


class Retriever:
    def __init__(self, vec, lsa, nn, pairs):
        self.vec, self.lsa, self.nn, self.pairs = vec, lsa, nn, pairs

    # -- build / load -------------------------------------------------------
    @classmethod
    def build(cls, openers_only: bool = True) -> "Retriever":
        rows = [json.loads(l) for l in PAIRS.open(encoding="utf-8") if l.strip()]
        if openers_only:
            rows = [r for r in rows if r.get("is_opener")]
        if not rows:
            raise SystemExit("no pairs - run build_pairs.py first")
        print(f"indexing {len(rows):,} pairs")

        texts = [r["customer"] for r in rows]
        vec = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            min_df=3, max_df=0.4, sublinear_tf=True, max_features=60_000,
        )
        X = vec.fit_transform(texts)
        n_comp = min(N_COMPONENTS, X.shape[1] - 1)
        lsa = make_pipeline(TruncatedSVD(n_components=n_comp, random_state=SEED),
                            Normalizer(copy=False))
        Xr = lsa.fit_transform(X)
        nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Xr)
        print(f"  tf-idf {X.shape[1]:,} terms -> LSA {Xr.shape[1]} dims")
        return cls(vec, lsa, nn, rows)

    def save(self, path: Path = INDEX) -> None:
        with path.open("wb") as fh:
            pickle.dump({"vec": self.vec, "lsa": self.lsa,
                         "nn": self.nn, "pairs": self.pairs}, fh)
        print(f"saved: {path} ({path.stat().st_size/1e6:.1f} MB)")

    @classmethod
    def load(cls, path: Path = INDEX) -> "Retriever":
        if not path.exists():
            r = cls.build()
            r.save(path)
            return r
        with path.open("rb") as fh:
            d = pickle.load(fh)
        return cls(d["vec"], d["lsa"], d["nn"], d["pairs"])

    # -- query --------------------------------------------------------------
    def search(self, text: str, k: int = 5) -> list[Precedent]:
        Xr = self.lsa.transform(self.vec.transform([text]))
        k = min(k, len(self.pairs))
        dist, idx = self.nn.kneighbors(Xr, n_neighbors=k)
        out = []
        for d, i in zip(dist[0], idx[0]):
            p = self.pairs[int(i)]
            # sklearn cosine 'distance' is 1 - cosine similarity
            out.append(Precedent(p["customer"], p["reply"], float(1.0 - d)))
        return out

    def grounding(self, text: str) -> float:
        """Top-1 cosine similarity - feeds escalation.py gate 4."""
        hits = self.search(text, k=1)
        return hits[0].score if hits else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("query", nargs="*", help="message to look up")
    args = ap.parse_args()

    if args.rebuild and INDEX.exists():
        INDEX.unlink()
    r = Retriever.load()

    queries = [" ".join(args.query)] if args.query else [
        "my parcel says delivered but i never got it",
        "i want to cancel my order it hasnt shipped yet",
        "charged twice for prime membership this month",
        "the blender arrived with a cracked jug",
        "asdkjh random gibberish nothing to do with anything",
    ]
    for q in queries:
        print("\n" + "=" * 76)
        print(f"QUERY: {q}")
        g = r.grounding(q)
        gate = "PASS" if g >= 0.60 else "FAIL -> escalate (no precedent)"
        print(f"grounding: {g:.3f}  [gate 4: {gate}]")
        print("=" * 76)
        for i, p in enumerate(r.search(q, k=args.k), 1):
            print(f"\n  [{i}] score={p.score:.3f}")
            print(f"      Q: {p.customer[:110]}")
            print(f"      A: {p.reply[:110]}")


if __name__ == "__main__":
    main()
