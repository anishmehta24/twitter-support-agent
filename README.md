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

## Report

Numbers below are from `results/RESULTS.md`, scored against round-2 labels.
**No human has labelled anything yet** (see [Golden evaluation set](#golden-evaluation-set));
treat every figure as agreement with a model until round 3 exists.

### 1. Problem framing

**What "good" means for AmazonHelp.** A public tweet reply cannot see the
account, cannot move money, and is read by everyone. So "good" is not
"resolves the case" - almost nothing resolvable happens in this channel - it
is: (a) never auto-handle something that needed a human, (b) when it does
reply, say what Amazon would actually have said, and (c) be honest about how
often (a) fails. I therefore optimise for a **cost-weighted escalation
error** (a false auto-handle costs 10x a needless escalation; the ratio is
stated in `escalation.py`, not implied) before reply quality, and I treat the
intent label as an input to that decision rather than as the product.

**What I chose not to build.** No fine-tuned classifier (250 labels is an
evaluation set, not a training set). No embedding retrieval (TF-IDF + LSA
runs on CPU in seconds and the grounding score it produces is the thing the
policy needs; swapping in sentence-transformers is a one-function change and
a legitimate follow-up experiment). No multi-turn handling: only conversation
openers are classified and indexed, because mid-thread turns ("yes the second
one") assume context the system does not have. No Banking77 transfer - its
77 intents are a different domain and would have made the taxonomy look more
principled than the data supports.

### 2. Results against baselines

| Decision | Trivial baseline | Simple baseline | System |
|---|---|---|---|
| Intent (macro-F1, weighted) | majority class **0.04** | keyword rules **0.50** | LLM **0.77** (acc 0.77; 0.84 on rows neither annotator flagged) |
| Escalation (cost @ 10:1, 250 rows) | escalate everything **131** | - | four-gate policy **500** (442 with gold intent) |
| Reply quality (judge, 1-5) | canned reply **2.36** | nearest historical reply **4.78** | grounded LLM **4.69** |

Two of three rows go the wrong way, and that is the result.

- **The escalation policy loses to "escalate everything."** With a 10:1 cost
  ratio, 43 false auto-handles (430) outweigh the 104 tickets it kept off a
  human's queue. Feeding it the *gold* intent only improves it to 442, so
  this is the policy, not the classifier: `AUTO_OK` permits
  `refund_or_return` and `delivery_not_received`, and both annotators say
  those usually need account access (9 and 7 false-autos respectively).
  The honest headline is: **as configured, the agent should not be switched
  on**; the auto-rate needed to break even at 10:1 is far higher than the
  precision of the gates allows.
- **The nearest-neighbour reply beats the LLM.** Returning Amazon's actual
  reply to the most similar past message scores 4.78; composing a new one
  from the top-5 precedents scores 4.69. The support corpus is repetitive
  enough that verbatim recall is a very hard bar - as `reply.py` predicted
  before any of this was measured. The LLM wins only on non-English (4.80 vs
  4.73) and fragments (4.77 vs 4.47), where the nearest precedent is a weak
  match.
- **The LLM classifier is near the ceiling.** 0.77 accuracy against a
  ceiling of ~0.63-0.84 (13 rows multi-intent, 79 flagged unsure by round 2;
  0.84 on the unflagged rows). Most remaining error is taxonomy, not model
  (see failure 3).

### 3. Failure analysis - top five

1. **Grounded replies address the customer by a previous customer's name.**
   The nearest-neighbour tier returns *"I'm sorry you're having a poor
   experience with our support, **Grace**."* to a customer who is not Grace
   (t10211); *"Thank you for your feedback, **Anand**."* (t10466). A crude
   scan finds dozens of such rows in the nearest-neighbour tier and a
   handful in the LLM tier even after the prompt forbids it. The judge gave
   these 4-5/5. *Hypothesis:* names are not in the retrieval signal and the
   judge rubric never asks "is the name right"; fix is a post-filter that
   strips any capitalised token absent from the incoming message, and a
   rubric line for it.
2. **Hard triggers fire on vocabulary, not situations.** 9 of 131 gold-auto
   rows were escalated by a hard trigger: `#Fraud` used as a hashtag on a
   late-delivery rant (t12200), *"who do we send scam emails to?"* (t15079 -
   a how-to with a documented public answer), *"thank you for the quick
   resolution and response to my account being compromised"* (t2069 -
   praise), and `\bpress\b` in *"press play"* matching the media trigger
   (t566). Earlier passes already removed `fire` (Fire TV) and `shock`
   (shockingly); this is the same failure mode with the next layer of words.
   *Hypothesis:* keyword triggers have a floor on precision; the right shape
   is a small classifier over the trigger *plus* intent, or at minimum
   negation/hashtag handling.
3. **`delivery_delayed` vs `order_cancel_or_change` is a taxonomy bug, not a
   model bug.** Five of the LLM's confusions are *"why hasn't my order been
   shipped yet"*, *"is it normal my order is 2 days in preparing shipment"* -
   labelled delayed by both annotators, predicted cancel/change by the model.
   The taxonomy text for `order_cancel_or_change` says *"or query an order's
   status pre-delivery"*, which is exactly these messages. The two annotators
   disagreed on 43/250 intents (kappa 0.80); the pairs they disagreed on are
   `item_damaged` vs `other` (6), `delivery_delayed` vs `complaint` (5),
   `other` vs `complaint` (5). *Hypothesis:* fix the one sentence, relabel,
   expect ~2 points of macro-F1 back.
4. **The judge does not enforce its own rubric.** The canned reply *"Sorry
   for the trouble! Please reach out to us here"* was scored `actionable=1`
   on 200 of 250 rows, although the rubric says "contact us" alone is not
   actionable. Its rationales praise tone. *Hypothesis:* the judge reads
   politeness as progress; with the human check outstanding, every reply
   score in this report should be read as an upper bound, and the 94%
   "send-ready" figure for nearest-neighbour is the single most inflated
   number in the results.
5. **The language gate escalates messages the system could have answered.**
   17-22 rows escalate purely for being non-English, but the grounding
   corpus contains German, French and Spanish replies and retrieval finds
   them (a German account-recovery question gets Amazon.de's actual German
   answer). *Hypothesis:* the gate was written before the corpus was
   measured; it should be per-language grounding, not a blanket rule.

### 4. What is misleading about my headline number

If I had to quote one number it would be *"0.77 macro-F1 and 4.7/5 reply
quality"*, and both are misleading in specific ways:

- **Nobody human has labelled the golden set.** Both rounds are models. Their
  agreement (kappa 0.80 on intent, **0.59 on the auto/escalate action**) is
  label reliability between two models that were trained on similar data
  and share blind spots; it is not evidence that either matches a support
  lead. The action label - the one that carries the 10:1 cost - is the one
  they agree on least.
- **The reply judge is unvalidated and demonstrably lenient** (failure 4).
  4.7/5 measures "sounds like Amazon", not "helps the customer".
- **The eval set is not the population.** Hard triggers are 16% of the set
  and 2% of traffic; the `weight` column corrects for this and the
  weight-corrected numbers are the ones in the tables, but per-stratum
  accuracy on 30-40 rows has wide intervals I have not computed.
- **The classifier under test and one of the annotators are sibling
  models** (Gemini 3.1 Flash-Lite classifies; Gemini 3.5 Flash/Flash-Lite
  labelled round 1). Scoring against round 2 (Claude) reduces but does not
  remove that circularity.
- **Reply quality is measured on all 250 messages, including the ones the
  policy would escalate.** In production the LLM only speaks on the 42% it
  auto-handles; its score on that subset is not separately reported.
- **"Grounded in the brand's history" inherits the brand's history.**
  Amazon's public replies are overwhelmingly polite redirects. A generator
  that reproduces them faithfully scores 0.98 on "grounded" while resolving
  nothing, and no automated metric in this repo can see that.

### 5. With one more week

1. Human round 3 on 50 rows and 40 judged replies - two hours of work that
   turns every caveat above into a number.
2. Retune the policy against the measured cost: drop `refund_or_return` and
   `delivery_not_received` from `AUTO_OK`, re-run, and report the cost curve
   across auto-rate rather than a single operating point.
3. Fix the taxonomy sentence (failure 3), the name post-filter (failure 1),
   and make the language gate per-language grounding (failure 5). Each is a
   one-line change with a cached, 25-second re-evaluation.
4. Split the reply evaluation by policy decision, so the LLM is scored only
   where it would actually speak.
5. Embedding retrieval as a controlled comparison against TF-IDF + LSA,
   reported as grounding-score AUC against the escalate/auto labels.

## Decision log

- **AmazonHelp over AppleSupport** despite Apple having 2.5x the
  conversations: 52.5% of Apple's replies are DM deflections, which are
  worthless as grounding material. Measured with `deflection.py` before
  choosing.
- **Escalation is the default and must be earned** through four ordered
  gates, because the cost is asymmetric. The 10:1 ratio is a stated
  assumption, not a fit.
- **Hard triggers override confidence.** A 0.99-confident `delivery_delayed`
  containing "someone used my card" still escalates.
- **Grounding score is an escalation gate**, not just a retrieval by-product:
  no comparable precedent means the agent would be improvising, which is
  exactly when it should not speak.
- **Only openers are classified and indexed.** Mid-thread turns assume
  context the system does not have at query time.
- **Two delivery intents, not one.** "Still waiting" and "marked delivered
  but missing" share vocabulary and have different resolution paths.
- **Language is a confound, not an intent.** Three of fifteen clusters were
  languages; they are recorded as a separate field and the taxonomy is
  language-independent.
- **Stratified, weighted golden set** rather than a uniform sample: a
  uniform 250 would hold ~5 hard triggers and the escalation policy would be
  unevaluable. Every row carries `population_share / sample_share`.
- **`multi` and `unsure` flags are reported, not cleaned.** They are the
  ceiling on what any single-label classifier can be shown to achieve.
- **Annotation guide written before any labelling**, and the identical text
  is the LLM annotator's system prompt, so every annotator - model or human -
  is held to the same tie-break rules.
- **Two independent model annotation rounds with provenance per row**, then
  a blind human spot-check, rather than presenting model labels as
  hand-labelled. The kappa between rounds is reported as label reliability.
- **Judge is a different model from the generator** (3.5 Flash-Lite judges
  3.1 Flash-Lite), and the judge prompt contains no example numbers - a small
  model copied the example `0,1,0,1,3` verbatim for every reply until it was
  removed. A degenerate-judge detector now flags identical score vectors.
- **Trivial and nearest-neighbour baselines built and scored before the
  LLM tier**, so its number had a denominator on the day it was measured.
- **All LLM outputs cached by message id and committed.** `results/`
  regenerates in ~25 seconds with no API key; the full run is ~450 calls.
- **Pure-Python streaming over the 2.8M-row CSV, no pandas.** The whole
  pipeline fits in well under 1 GB, which mattered on the 7 GB machine it was
  built on.

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
