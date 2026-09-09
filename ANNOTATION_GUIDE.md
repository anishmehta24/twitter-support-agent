# Annotation guide — golden evaluation set

This is the spec the golden set is labelled against. It is deliberately written
before labelling starts, so decisions are made once and applied consistently
rather than invented per-message. The same text is given to the LLM labeller,
so human and model are judged against the same definitions.

## How the sample was drawn

250 messages from 17,590 unique AmazonHelp conversation openers, seeded
(`SEED = 20260910`) and reproducible via `sample_golden.py`.

Four **disjoint** strata, assigned by priority so a French fraud report counts
once, as `hard_trigger`:

| Stratum | Population | Pop% | Sampled | Samp% | Weight |
|---|---|---|---|---|---|
| hard_trigger | 361 | 2.1% | 40 | 16.0% | 0.128 |
| non_english | 1,704 | 9.7% | 30 | 12.0% | 0.807 |
| short_or_fragment | 2,005 | 11.4% | 30 | 12.0% | 0.950 |
| core | 13,520 | 76.9% | 150 | 60.0% | 1.281 |

Rare strata are oversampled on purpose. Hard triggers are 2.1% of the corpus,
so a uniform sample of 250 would contain ~5 of them and the escalation policy
would be unevaluable. The cost is that **raw accuracy over this set is not
population accuracy** — hence the `weight` column
(`population_share / sample_share`). Report per-stratum metrics *and*
weight-corrected overall metrics; reporting only the unweighted overall number
would overstate performance on the rare, expensive cases.

The sample is shuffled before annotation so no stratum is labelled in a block,
which would let fatigue or drift correlate with stratum.

## Labels

Two per message, because the system makes two decisions.

### 1. Intent — exactly one

| Label | Use when |
|---|---|
| `delivery_delayed` | Still waiting. Late, or will miss a promised date. |
| `delivery_not_received` | Tracking says **delivered** but the customer has nothing. Includes stolen/misdelivered. |
| `item_damaged_wrong_missing` | Physically arrived but damaged, defective, wrong item, or incomplete. |
| `order_cancel_or_change` | Cancel/modify pre-delivery, or Amazon cancelled it unexpectedly. |
| `refund_or_return` | Money owed back, refund delayed, or how do I return this. |
| `payment_or_charge` | Card declined, duplicate charge, gift card, Amazon Pay. **Not Prime fees.** |
| `prime_membership` | Prime signup, charge, cancellation, benefits, Prime Video. |
| `account_access` | Locked out, hacked, password, verification codes, closure. |
| `service_complaint_or_feedback` | General rant **or praise** with no single actionable case. |
| `other` | Unintelligible, fragment, or not a support request. |

### 2. Action — `auto_handle` or `escalate`

Label what a **competent support org should do**, not what you predict the
model will do. You are creating ground truth, not imitating the system.

Escalate when any of these hold:
- financial crime, account compromise, legal threat, physical safety, or a
  vulnerable customer is indicated
- resolving it requires verified account access or moving money
- the message is too ambiguous to answer safely
- a wrong public reply would be materially harmful

Auto-handle when a grounded, generic reply genuinely advances the case — status
guidance, documented return process, acknowledging praise.

## Tie-break rules

Written in advance so ambiguity is resolved the same way every time.

1. **Multiple problems → label the one the customer leads with**, and set the
   `multi` flag. Leading position is the tie-break because it is objective;
   "most severe" is not, and would drift between rounds.
2. **`delivery_delayed` vs `delivery_not_received`** turns on one question: does
   tracking claim delivery? If yes → `not_received`. If still in transit or
   simply late → `delayed`. This pair is the most confusable in the taxonomy.
3. **Prime fees are `prime_membership`**, never `payment_or_charge`, even when
   phrased as a bank charge.
4. **A complaint *about* a specific case is that case's intent**, not
   `service_complaint_or_feedback`. Use the complaint label only when no
   specific actionable case is attached.
5. **Label the intent regardless of language.** Non-English messages get a real
   intent label; language is recorded separately and is not a category.
6. **When you would not bet on it, set `unsure`** rather than agonising. That
   flag is data, not failure — it marks where the taxonomy is genuinely weak.

## Flags

- `multi` — more than one distinct problem in the message
- `unsure` — you would not bet on this label

Neither is noise to be cleaned up. `multi` is a hard ceiling on what any
single-label classifier can achieve; `unsure` marks where the taxonomy needs
work. Both are reported.

## Two-round protocol

Single annotator, so inter-annotator agreement is impossible. The honest
substitute is **intra-annotator agreement**:

```bash
python annotate.py                     # round 1
# wait at least a day
python annotate.py --round 2 --blind   # same messages, reshuffled, answers hidden
python agreement.py                    # Cohen's kappa + confusable pairs
```

Round 2 reshuffles with a different seed so position cannot cue recall, and
`--blind` hides round-1 answers.

Cohen's kappa is used rather than raw agreement because raw agreement is
inflated by class imbalance — always guessing the majority intent scores well
on percentage agreement and ~0 on kappa.

**This number is the ceiling on measurable model performance.** If self-kappa
is 0.72, a classifier scoring 0.90 against these labels is not measuring what
it appears to. Quote it next to every headline metric.
