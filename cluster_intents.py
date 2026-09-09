# -*- coding: utf-8 -*-
"""Derive a candidate intent taxonomy from AmazonHelp openers.

Pipeline: TF-IDF -> TruncatedSVD (LSA) -> L2 normalize -> KMeans.

The SVD step is not optional. Clustering raw TF-IDF of short texts collapses
into one mega-cluster plus singletons, because tweet vectors are so sparse that
nearly every pair is orthogonal and there is no geometry for centroids to find.
Projecting to a dense latent space restores it.

Clusters are a starting point for human naming, not the final taxonomy.
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

SRC = Path(r"C:\Users\BIT\oss\hiver\data\amazon_openers.tsv")
OUT = Path(r"C:\Users\BIT\oss\hiver\data\amazon_clusters.tsv")
SEED = 42
N_COMPONENTS = 100
random.seed(SEED)
np.random.seed(SEED)
csv.field_size_limit(10_000_000)

texts = []
with SRC.open("r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        c = (row.get("clean") or "").strip()
        if len(c) >= 15:
            texts.append(c)
print(f"loaded {len(texts):,} openers")

EXTRA_STOP = {
    "amazon", "amazonhelp", "hi", "hello", "hey", "please", "pls", "thanks",
    "thank", "im", "ive", "dont", "cant", "just", "got", "get", "amp", "gt",
    "lt", "https", "co", "like", "know", "want", "need", "help", "guys",
}
vec = TfidfVectorizer(
    lowercase=True, stop_words=list(EXTRA_STOP | {"a", "the", "and"}) + ["english"],
    ngram_range=(1, 2), min_df=10, max_df=0.30,
    max_features=30_000, sublinear_tf=True,
)
# use sklearn's english list too
vec.set_params(stop_words="english")
X = vec.fit_transform(texts)
terms = vec.get_feature_names_out()
print(f"tf-idf: {X.shape[0]:,} docs x {X.shape[1]:,} terms")

svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
lsa = make_pipeline(svd, Normalizer(copy=False))
Xr = lsa.fit_transform(X)
print(f"LSA -> {Xr.shape[1]} dims, explained variance "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}%\n")

print("k sweep (silhouette, cosine, 4k sample):")
sample = np.random.choice(Xr.shape[0], size=min(4000, Xr.shape[0]), replace=False)
best_k, best_s = None, -1.0
for k in range(6, 17):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=8)
    lab = km.fit_predict(Xr)
    s = silhouette_score(Xr[sample], lab[sample], metric="cosine", random_state=SEED)
    mark = ""
    if s > best_s:
        best_s, best_k, mark = s, k, "  <-- best"
    sizes = np.bincount(lab, minlength=k)
    print(f"  k={k:>2}  silhouette={s:6.4f}   largest cluster={100*sizes.max()/len(lab):5.1f}%{mark}")

K = int(sys.argv[1]) if len(sys.argv) > 1 else best_k
print(f"\n=== final clustering, k={K} ===\n")
km = KMeans(n_clusters=K, random_state=SEED, n_init=15)
labels = km.fit_predict(Xr)

# map centroids back to term space for interpretability
centroids_term = svd.inverse_transform(km.cluster_centers_)
order = centroids_term.argsort()[:, ::-1]
sizes = np.bincount(labels, minlength=K)

for c in np.argsort(-sizes):
    top = [terms[i] for i in order[c, :12]][:10]
    print(f"--- cluster {c}  n={sizes[c]:,} ({100*sizes[c]/len(labels):.1f}%) ---")
    print("  terms: " + ", ".join(top))
    members = np.where(labels == c)[0]
    for i in random.sample(list(members), min(5, len(members))):
        print(f"    - {' '.join(texts[i].split())[:145]}")
    print()

with OUT.open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["cluster", "text"])
    for lab, t in zip(labels, texts):
        w.writerow([int(lab), t])
print(f"written: {OUT}")
