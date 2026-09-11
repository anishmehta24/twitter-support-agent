# Twitter Support Agent

Intent classification, grounded reply drafting, and escalation routing over the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (~2.8M tweets).

**Status: work in progress.** The full pipeline and evaluation harness are
built and have been run end to end; see [`results/RESULTS.md`](results/RESULTS.md).
**Those numbers are against model-annotated labels** (two independent model
rounds, intent kappa 0.80 between them; results are scored against round 2).
The blind human spot-check (`annotate.py --round 3`) and the human check on the
judge (`judge.py --human`) have not been done yet. Until they are, every number
in `results/` measures agreement with a model, not with a person.

---

## Quickstart

```bash
./scripts/fetch_data.sh                  # 516 MB, not committed
python -m venv .venv && .venv/Scripts/activate
pip install scikit-learn
cp .env.example .env                     # optional: add an API key

python brand_stats.py                    # brand volume + thread shape
python deflection.py                     # how substantive is each brand's support?
python extract_openers.py                # -> data/amazon_openers.tsv
python cluster_intents.py                # -> data/amazon_clusters.tsv, taxonomy evidence
python sample_golden.py                  # -> data/golden_pool.jsonl (250, stratified + weighted)
python escalation.py                     # policy demo over worked cases

# Golden set: two independent model passes, then a blind human spot-check
python label_golden.py                   # -> data/golden_r1.jsonl (Gemini, annotator recorded)
#   data/golden_r2.jsonl                 # second pass by a different model (Claude), committed
python annotate.py --round 3 --blind     # -> data/golden_r3.jsonl (you, 50-row blind spot-check)
python agreement.py                      # kappa: r1 vs r2, and r3 vs each

python build_pairs.py                    # -> data/amazon_pairs.jsonl (grounding corpus)
python agent.py "my parcel says delivered but i never got it"   # end to end

# Evaluation - score against the HUMAN round, not the LLM one
python evaluate.py --gold data/golden_r2.jsonl --skip-replies    # intent + escalation, no LLM, ~1 min
python evaluate.py --gold data/golden_r2.jsonl --model gemini-3.1-flash-lite --judge-model gemini-3.5-flash-lite
#   ^ ~25s from the committed caches; ~450 calls and ~1 h on the Gemini free tier from scratch
python judge.py --human --n 40           # score a blind subset yourself
python judge.py --agreement              # judge vs human kappa
```

LLM backend is auto-detected: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` (free tier is enough for a full run), then a local Ollama
(`ollama pull qwen2.5:3b`). Keys can live in a gitignored `.env`. Without
any, `agent.py` and `evaluate.py` fall back to the keyword classifier and
skip reply drafting. All LLM outputs are cached by message id under `data/`
so a re-run never pays twice.

Everything is seeded. `data/golden_pool.jsonl` is committed so the evaluation
set is pinned and identical across runs.

---

## Why AmazonHelp

Brand choice was made on measured data, not volume. The metric that decided it
is **how often a brand actually resolves in public** — a reply saying "please
DM us" is worthless as grounding material, because the resolution happened in a
channel the dataset never captured.

| Brand | Replies | Deflect% | **Substantive%** | Conversations |
|---|---|---|---|---|
| hulu_support | 21,872 | 0.5% | 97.8% | 7,209 |
| **AmazonHelp** | **169,840** | **0.6%** | **94.7%** | **17,863** |
| SouthwestAir | 28,977 | 16.9% | 77.8% | 18,520 |
| Delta | 42,253 | 16.5% | 73.5% | 20,957 |
| AppleSupport | 106,860 | **52.5%** | 47.4% | 44,479 |
| TMobileHelp | 34,317 | **81.9%** | 17.8% | 6,039 |

AppleSupport has the most conversations of any brand and is the obvious pick on
volume — but **52.5% of its replies are DM deflections**. AmazonHelp gives
~161,000 substantive replies, 3× the next best, and its 9.5 replies per
conversation are genuine end-to-end resolution arcs rather than deflection
ping-pong.

Rejected alternatives: `hulu_support` (cleanest data anywhere, but only 7,209
conversations and low stakes variance, leaving the escalation work thin);
`Delta` (most bounded intent space, but 26% deflection + contentless apology).

## Intent taxonomy

Derived from 17,590 unique conversation openers via TF-IDF → LSA → k-means
(k chosen by silhouette sweep over 6–16; peak 0.167 at k=15).

`delivery_delayed` · `delivery_not_received` · `item_damaged_wrong_missing` ·
`order_cancel_or_change` · `refund_or_return` · `payment_or_charge` ·
`prime_membership` · `account_access` · `service_complaint_or_feedback` ·
`other`

Two findings that shaped it:

- **3 of 15 clusters are languages, not intents.** ~11.7% of openers are
  French, Spanish or German, and lexical clustering separates language *before*
  intent. Language is a confound, not a category.
- **Delivery splits in two.** "Still waiting" and "marked delivered but missing"
  share vocabulary but have completely different resolution paths, so they are
  separate intents.

Honest limits: LSA explains only 22.3% of variance, silhouette peaks at a
modest 0.167, and a 31% residual cluster resists lexical separation. These
clusters are *evidence for* a taxonomy, not the taxonomy itself.

## Escalation policy

Framed as a decision under asymmetric cost, not a classification: wrongly
auto-handling a fraud report costs far more than a needless escalation. The
cost ratio is stated explicitly in code (10:1), not implied. Escalation is the
default; a message must clear four gates, each of which names itself:

1. **Hard trigger** — fraud, account compromise, legal, safety, vulnerability,
   media. Overrides everything, including high confidence.
2. **Intent policy** — can a public tweet reply resolve this intent at all?
3. **Confidence** — classifier certainty ≥ 0.75.
4. **Grounding** — retrieval found a real precedent (cosine ≥ 0.60). Without
   one the agent is improvising, which is exactly when it should not be
   unsupervised.

### A false positive worth reporting

The first pass fired on 3.7% of messages. Inspecting matches showed `\bfire\b`
was catching **Fire TV, Fire Stick and Fire Kids** — Amazon's own hardware —
while `shock\w*` matched "shockingly" and `not mine` matched misdelivered
parcels. After requiring hazard context, `safety_or_harm` fell **295 → 16** and
the overall rate settled at **2.1%**.

## Golden evaluation set

250 examples, stratified and **weighted**. Labelled against the same written
guide (`ANNOTATION_GUIDE.md`) in independent rounds, each row recording its
`annotator`: round 1 by Gemini (`label_golden.py`), round 2 by Claude, and a
**blind human spot-check** as round 3 (`annotate.py --round 3 --blind`, 50
rows). `agreement.py` reports Cohen's kappa between every pair. Results are
scored against round 2; the model-vs-model kappa is label reliability, the
human-vs-model kappa on the subset is the number that says whether the models
can be trusted as annotators at all. Rare strata are deliberately
oversampled — hard triggers are 2.1% of the corpus, so a uniform sample would
draw ~5 and leave the escalation policy unevaluable.

| Stratum | Population | Pop% | Sampled | Weight |
|---|---|---|---|---|
| hard_trigger | 361 | 2.1% | 40 | 0.128 |
| non_english | 1,704 | 9.7% | 30 | 0.807 |
| short_or_fragment | 2,005 | 11.4% | 30 | 0.950 |
| core | 13,520 | 76.9% | 150 | 1.281 |

Because of this, **unweighted accuracy over the golden set is not population
accuracy**. Every row carries `weight = population_share / sample_share`, and
results are reported both per-stratum and weight-corrected.

## Retrieval and reply generation

`build_pairs.py` joins inbound tweets to AmazonHelp's actual replies via
`response_tweet_id`, producing **77,972 pairs, of which 16,606 are openers**.
Only openers are indexed: mid-thread turns assume context the retriever will
not have at query time ("yes, the second one"). Pleasantries are dropped —
*"Ok, thanks for your help"* → *"Not a problem!"* is real conversation carrying
no resolution.

`retrieve.py` indexes with TF-IDF → LSA → cosine nearest neighbours. It serves
two purposes, and the second is easy to miss: it supplies precedents to the
generator, **and** the grounding score that escalation gate 4 depends on.

```
QUERY: charged twice for prime membership this month
grounding: 0.928  [gate 4: PASS]
  [1] 0.928  Q: y'all charged me twice for my prime membership ...
             A: I'm sorry to hear this. We'd be happy to check on this with you here: ^AF

QUERY: asdkjh random gibberish nothing to do with anything
grounding: 0.440  [gate 4: FAIL -> escalate (no precedent)]
```

`reply.py` implements three tiers so the system has something to beat:

| Tier | Method | Needs an LLM |
|---|---|---|
| `trivial` | one canned reply for everything | no |
| `nearest_neighbour` | Amazon's actual historical reply, verbatim | no |
| `grounded_llm` | composed from top-k precedents | yes |

### Two findings already visible

**The nearest-neighbour baseline leaks personalisation.** Asked about a missing
parcel it returns *"I am sorry this has not turned up yet, **Emma**. Was this
marked as delivered today?"* — the previous customer's name, addressed to a new
one. A concrete failure mode, not a hypothetical.

**The grounding corpus is mostly soft redirects.** Even after filtering explicit
DM deflections, Amazon's public replies overwhelmingly acknowledge and then move
the conversation elsewhere: *"please reach out here"*, *"give us a ring or chat
here"*. A generator grounded in these learns to deflect politely, and will score
well against a judge rewarding tone and groundedness **while resolving nothing**.
No automated metric in this repo can see that.

## Layout

```
brand_stats.py       streaming volume/thread stats per brand
deflection.py        DM-deflection and substantiveness per brand
extract_openers.py   conversation openers -> TSV
cluster_intents.py   TF-IDF -> LSA -> k-means, taxonomy evidence
llm.py               multi-provider LLM client (Anthropic / OpenAI / Ollama)
label_intents.py     LLM labelling, batched + resumable + token accounting
escalation.py        four-gate auto-vs-escalate policy with stated reasons
build_pairs.py       (customer -> Amazon reply) grounding corpus
retrieve.py          TF-IDF -> LSA -> cosine retrieval; supplies grounding score
reply.py             three reply tiers: trivial / nearest-neighbour / grounded LLM
classify.py          three intent classifiers: majority / keyword / LLM
label_golden.py      LLM first-pass labels for the golden set (round 1)
agent.py             end to end: classify -> retrieve -> decide -> draft
sample_golden.py     stratified + weighted golden-set sampler
annotate.py          keyboard-driven human annotation, two rounds
agreement.py         Cohen's kappa self-agreement + confusable pairs
judge.py             LLM-as-judge rubric, blind human scoring, judge-vs-human kappa
evaluate.py          the harness: intent F1, escalation cost, reply quality -> results/
scripts/fetch_data.sh
```

All heavy passes stream the CSV row by row and hold only counters — the full
2.8M-row scan runs in well under 1 GB of RAM, with no pandas dependency.

## Evaluation harness

`evaluate.py` scores the three decisions separately, because they fail
separately:

| Section | What is measured | Baselines |
|---|---|---|
| Intent | accuracy and macro-F1, raw and weight-corrected, per stratum; plus a "clean" accuracy over rows the annotator did not flag `multi`/`unsure` | majority class, keyword rules |
| Escalation | false-auto and false-escalate counted separately and combined at the stated 10:1 cost; run once with the predicted intent and once with the gold intent so classifier error and policy error are not confused | escalate-everything, auto-everything |
| Replies | four binary rubric checks (grounded, safe, actionable, on-tone) + overall 1-5 from an LLM judge, per tier | trivial canned reply, nearest-neighbour verbatim |

The judge is validated, not trusted: `judge.py --human` shows you a blind,
tier-hidden subset to score against the same rubric, and `--agreement`
reports per-criterion Cohen's kappa and the judge's leniency bias. Until that
has been run, every judge number in `results/` is labelled unvalidated.

## Not done yet

- **Blind human spot-check of the golden set** (`annotate.py --round 3 --blind`,
  50 rows) - two model rounds exist and agree at kappa 0.80 on intent, but no
  person has labelled anything yet
- Human scoring of ~40 replies for judge validation
- Report: results vs baselines, top-5 failure modes, "what is misleading about
  my headline number", decision log
