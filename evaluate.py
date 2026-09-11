# -*- coding: utf-8 -*-
"""Evaluation harness. Produces every number the report quotes.

    python evaluate.py                       # everything, cached, resumable
    python evaluate.py --skip-replies        # intent + escalation only (no LLM)
    python evaluate.py --limit 60            # subsample for the 15-minute budget
    python evaluate.py --classifier keyword  # which classifier drives the policy

Three sections, matching the three decisions the agent makes:

  1. INTENT      macro-F1 and accuracy for majority / keyword / llm, raw and
                 weight-corrected, per stratum, with the ceiling implied by
                 the annotator's own `multi` and `unsure` flags.
  2. ESCALATION  the policy scored against gold `action`. False-auto (the
                 expensive error) and false-escalate counted separately and
                 combined with the stated 10:1 cost ratio. Run twice: with the
                 predicted intent and with the gold intent, so classifier
                 error and policy error are not confused.
  3. REPLIES     three tiers judged by the rubric in judge.py. Judge-vs-human
                 agreement is reported alongside if data/judge_human.jsonl
                 exists; without it the judge scores are unvalidated.

Weighting. Every golden row carries weight = population_share / sample_share
(see sample_golden.py). "Raw" metrics treat the 250 rows equally and
over-represent hard triggers 7x. "Weighted" metrics are what production would
see. Both are printed; the report must explain the gap.

Outputs: results/results.json (machine) and results/RESULTS.md (human).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from classify import make
from escalation import COST_FALSE_AUTO, COST_FALSE_ESCALATE, decide, expected_cost
from judge import CRITERIA, Judge
from llm import utf8_console
from reply import ReplyGenerator
from retrieve import Retriever

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
GOLD_R1 = DATA / "golden_r1.jsonl"
TIERS = ["trivial", "nearest_neighbour", "grounded_llm"]
INTENTS = [
    "delivery_delayed", "delivery_not_received", "item_damaged_wrong_missing",
    "order_cancel_or_change", "refund_or_return", "payment_or_charge",
    "prime_membership", "account_access", "service_complaint_or_feedback", "other",
]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def macro_f1(gold: list[str], pred: list[str], w: list[float] | None = None) -> float:
    w = w or [1.0] * len(gold)
    tp, fp, fn = defaultdict(float), defaultdict(float), defaultdict(float)
    for g, p, wi in zip(gold, pred, w):
        if g == p:
            tp[g] += wi
        else:
            fp[p] += wi
            fn[g] += wi
    f1s = []
    for c in set(gold):                     # classes absent from gold are undefined, skipped
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def accuracy(gold, pred, w=None) -> float:
    w = w or [1.0] * len(gold)
    return sum(wi for g, p, wi in zip(gold, pred, w) if g == p) / sum(w) if w else 0.0


def per_class_f1(gold, pred) -> dict[str, tuple[float, int]]:
    out = {}
    for c in INTENTS:
        tp = sum(1 for g, p in zip(gold, pred) if g == p == c)
        fp = sum(1 for g, p in zip(gold, pred) if p == c and g != c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        n = tp + fn
        if n == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n
        out[c] = (2 * prec * rec / (prec + rec) if prec + rec else 0.0, n)
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def load_gold(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nThe golden set has not been labelled yet. Run "
            "`python annotate.py` (250 messages, ~2 hours), then come back.")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["id"])
    if limit:
        # Take a stratified slice so a subsample still has hard triggers in it.
        by_s = defaultdict(list)
        for r in rows:
            by_s[r["stratum"]].append(r)
        share = limit / len(rows)
        rows = [r for s, xs in sorted(by_s.items())
                for r in xs[: max(1, round(len(xs) * share))]]
    return rows


def eval_intent(gold: list[dict], names: list[str], client) -> tuple[dict, dict[str, list]]:
    texts = [r["text"] for r in gold]
    ids = [r["id"] for r in gold]
    g = [r["intent"] for r in gold]
    w = [r["weight"] for r in gold]
    out, preds_by = {}, {}
    print("\n" + "=" * 78 + "\n1. INTENT\n" + "=" * 78)
    print(f"{'classifier':<12}{'acc':>7}{'acc(w)':>8}{'F1':>7}{'F1(w)':>8}   "
          f"{'clean acc':>10}   per-stratum acc")
    for name in names:
        clf = make(name, gold_intents=g, client=client) if name == "llm" else make(name, gold_intents=g)
        preds = clf.predict(texts, ids=ids) if name == "llm" else clf.predict(texts)
        p = [x.intent for x in preds]
        preds_by[name] = preds
        # "Clean" = rows the annotator did not flag multi or unsure: the part of
        # the set where a single label is actually defensible.
        clean = [i for i, r in enumerate(gold) if not r.get("multi") and not r.get("unsure")]
        strata = sorted({r["stratum"] for r in gold})
        sacc = {}
        for s in strata:
            idx = [i for i, r in enumerate(gold) if r["stratum"] == s]
            sacc[s] = accuracy([g[i] for i in idx], [p[i] for i in idx])
        rec = {
            "accuracy": accuracy(g, p), "accuracy_weighted": accuracy(g, p, w),
            "macro_f1": macro_f1(g, p), "macro_f1_weighted": macro_f1(g, p, w),
            "accuracy_clean": accuracy([g[i] for i in clean], [p[i] for i in clean]),
            "n_clean": len(clean), "per_stratum_accuracy": sacc,
            "per_class_f1": {k: {"f1": f, "n": n} for k, (f, n) in per_class_f1(g, p).items()},
            "confusions": [
                {"gold": a, "pred": b, "n": n} for (a, b), n in
                Counter((a, b) for a, b in zip(g, p) if a != b).most_common(5)],
        }
        out[name] = rec
        print(f"{name:<12}{rec['accuracy']:>7.3f}{rec['accuracy_weighted']:>8.3f}"
              f"{rec['macro_f1']:>7.3f}{rec['macro_f1_weighted']:>8.3f}   "
              f"{rec['accuracy_clean']:>10.3f}   "
              + "  ".join(f"{s[:5]}={v:.2f}" for s, v in sacc.items()))
    n_multi = sum(1 for r in gold if r.get("multi"))
    n_unsure = sum(1 for r in gold if r.get("unsure"))
    print(f"\nceiling: {n_multi} multi-intent + {n_unsure} unsure of {len(gold)} "
          f"-> ~{100*(1 - (n_multi + n_unsure)/len(gold)):.0f}% is the most a single-label "
          "classifier can be shown to get right")
    best = names[-1]
    if out[best]["confusions"]:
        print(f"\ntop confusions ({best}):")
        for c in out[best]["confusions"]:
            print(f"  {c['n']:>3}x  gold={c['gold']}  pred={c['pred']}")
    return out, preds_by


def eval_escalation(gold: list[dict], preds, retriever: Retriever, label: str) -> dict:
    g = [r["action"] for r in gold]
    w = [r["weight"] for r in gold]
    p, gates = [], Counter()
    fa_examples = []
    for r, pr in zip(gold, preds):
        grounding = retriever.grounding(r["text"])
        d = decide(pr.intent, pr.confidence, r["text"], grounding, language=pr.language)
        p.append(d.action.value)
        gates[d.gate] += 1
        if r["action"] == "escalate" and d.action.value == "auto_handle" and len(fa_examples) < 5:
            fa_examples.append({"id": r["id"], "text": r["text"][:140], "gold_intent": r["intent"],
                                "pred_intent": pr.intent, "reason": d.reason})
    fa = sum(1 for a, b in zip(g, p) if a == "escalate" and b == "auto_handle")
    fe = sum(1 for a, b in zip(g, p) if a == "auto_handle" and b == "escalate")
    fa_w = sum(wi for a, b, wi in zip(g, p, w) if a == "escalate" and b == "auto_handle")
    fe_w = sum(wi for a, b, wi in zip(g, p, w) if a == "auto_handle" and b == "escalate")
    n_esc = g.count("escalate")
    tp = sum(1 for a, b in zip(g, p) if a == b == "escalate")
    prec = tp / p.count("escalate") if p.count("escalate") else 0.0
    rec = tp / n_esc if n_esc else 0.0
    auto_rate = p.count("auto_handle") / len(p)
    auto_rate_w = sum(wi for b, wi in zip(p, w) if b == "auto_handle") / sum(w)
    out = {
        "accuracy": accuracy(g, p), "accuracy_weighted": accuracy(g, p, w),
        "escalate_precision": prec, "escalate_recall": rec,
        "false_auto": fa, "false_escalate": fe,
        "false_auto_weighted": fa_w, "false_escalate_weighted": fe_w,
        "expected_cost": expected_cost(fa, fe),
        "expected_cost_weighted": expected_cost(fa_w, fe_w),
        "auto_rate": auto_rate, "auto_rate_weighted": auto_rate_w,
        "gates": dict(gates), "false_auto_examples": fa_examples,
    }
    print(f"\n[{label}]  acc {out['accuracy']:.3f} (w {out['accuracy_weighted']:.3f})   "
          f"escalate P {prec:.2f} / R {rec:.2f}   auto-rate {auto_rate:.2f} (w {auto_rate_w:.2f})")
    print(f"   false-auto {fa} (w {fa_w:.1f})   false-escalate {fe} (w {fe_w:.1f})   "
          f"cost {out['expected_cost']:.0f} (w {out['expected_cost_weighted']:.1f})"
          f"  @ {COST_FALSE_AUTO:.0f}:{COST_FALSE_ESCALATE:.0f}")
    print("   gates: " + ", ".join(f"{k}={v}" for k, v in gates.most_common()))
    return out


def eval_escalation_all(gold, preds_main, retriever) -> dict:
    print("\n" + "=" * 78 + "\n2. ESCALATION\n" + "=" * 78)
    g = [r["action"] for r in gold]
    n_esc = g.count("escalate")
    out = {"n_escalate_gold": n_esc, "n_auto_gold": len(g) - n_esc}
    # Two trivial policies bound the problem.
    fe_all = len(g) - n_esc
    out["escalate_everything"] = {"expected_cost": expected_cost(0, fe_all), "false_auto": 0,
                                  "false_escalate": fe_all}
    out["auto_everything"] = {"expected_cost": expected_cost(n_esc, 0), "false_auto": n_esc,
                              "false_escalate": 0}
    print(f"gold: {n_esc} escalate / {len(g) - n_esc} auto")
    print(f"[escalate everything]  cost {out['escalate_everything']['expected_cost']:.0f}   "
          f"[auto everything]  cost {out['auto_everything']['expected_cost']:.0f}")
    out["predicted_intent"] = eval_escalation(gold, preds_main, retriever, "policy + predicted intent")
    from classify import Prediction
    oracle = [Prediction(r["intent"], 1.0, r["language"]) for r in gold]
    out["gold_intent"] = eval_escalation(gold, oracle, retriever, "policy + GOLD intent (policy error only)")
    if out["predicted_intent"]["false_auto_examples"]:
        print("\nfalse-auto examples (gold says escalate, agent auto-handled):")
        for e in out["predicted_intent"]["false_auto_examples"]:
            print(f"  {e['id']}  gold={e['gold_intent']} pred={e['pred_intent']}\n"
                  f"      {e['text']}\n      -> {e['reason'][:90]}")
    return out


def eval_replies(gold: list[dict], retriever: Retriever, client, tiers: list[str],
                 judge_client=None) -> dict:
    print("\n" + "=" * 78 + "\n3. REPLIES\n" + "=" * 78)
    judge = Judge(client=judge_client or client)
    out = {}
    for tier in tiers:
        path = DATA / f"replies_{tier}.jsonl"
        have = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    have[r["id"]] = r
        todo = [r for r in gold if r["id"] not in have]
        if todo:
            try:
                gen = ReplyGenerator(retriever, tier=tier, client=client)
                with path.open("a", encoding="utf-8") as sink:
                    for i, r in enumerate(todo, 1):
                        o = gen.generate(r["text"])
                        rec = {"id": r["id"], "text": r["text"], "stratum": r["stratum"],
                               "weight": r["weight"], "reply": o["reply"],
                               "grounding": o["grounding"], "precedents": o["precedents"][:3],
                               # Provenance, so a cache row from a different model
                               # can be recognised and purged rather than mixed in.
                               "model": gen.client.model if tier == "grounded_llm" else None}
                        have[r["id"]] = rec
                        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if tier == "grounded_llm" and i % 10 == 0:
                            print(f"  drafted {i}/{len(todo)}  ({gen.client.usage})")
            except SystemExit as e:
                print(f"[{tier}] skipped: {e}")
                continue
        items = [have[r["id"]] for r in gold if r["id"] in have]
        try:
            scores = judge.score(items, DATA / f"judge_{tier}.jsonl")
        except SystemExit as e:
            print(f"[{tier}] judge skipped: {e}")
            continue
        rows = [(r, scores[r["id"]]) for r in items if r["id"] in scores]
        if not rows:
            continue
        w = [r["weight"] for r, _ in rows]
        rec = {"n": len(rows)}
        for c in CRITERIA + ["overall"]:
            vals = [s[c] for _, s in rows]
            rec[c] = sum(vals) / len(vals)
            rec[c + "_weighted"] = sum(v * wi for v, wi in zip(vals, w)) / sum(w)
        # A judge that gives every reply the same vector is not judging. Small
        # local models do this by echoing the output format; flag it loudly.
        distinct = {tuple(s[c] for c in CRITERIA + ["overall"]) for _, s in rows}
        rec["degenerate"] = len(rows) >= 5 and len(distinct) == 1
        if rec["degenerate"]:
            print(f"  WARNING [{tier}]: judge returned an identical score for all "
                  f"{len(rows)} replies - treat these numbers as invalid")
        rec["mean_len"] = sum(len(r["reply"]) for r, _ in rows) / len(rows)
        rec["pct_send_ready"] = 100 * sum(1 for _, s in rows if s["overall"] >= 4) / len(rows)
        out[tier] = rec
    if out:
        print(f"\n{'tier':<20}{'n':>5}" + "".join(f"{c:>12}" for c in CRITERIA)
              + f"{'overall':>9}{'ovl(w)':>8}{'send-ready':>12}")
        for t, r in out.items():
            print(f"{t:<20}{r['n']:>5}" + "".join(f"{r[c]:>12.2f}" for c in CRITERIA)
                  + f"{r['overall']:>9.2f}{r['overall_weighted']:>8.2f}{r['pct_send_ready']:>11.0f}%")
    return out


# ---------------------------------------------------------------------------
def write_results(res: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Results", "", f"Golden set: {res['n']} rows from `{res['gold_file']}`"
         + (f" (limit {res['limit']})" if res.get("limit") else ""),
         f"Classifier driving the policy: `{res['classifier']}`", ""]
    if res.get("annotators"):
        L += ["> **Gold labels are LLM-annotated** (" + ", ".join(
                  f"{k}: {v}" for k, v in res["annotators"].items())
              + "), not yet human-verified. Every number below is agreement with a "
              "model, not with a person, until `annotate.py --round 2 --blind` has "
              "been run and this is re-scored against `golden_r2.jsonl`.", ""]
    L += ["## Intent", "", "| classifier | acc | acc (w) | macro-F1 | macro-F1 (w) | clean acc |",
          "|---|---|---|---|---|---|"]
    for k, v in res["intent"].items():
        L.append(f"| {k} | {v['accuracy']:.3f} | {v['accuracy_weighted']:.3f} | "
                 f"{v['macro_f1']:.3f} | {v['macro_f1_weighted']:.3f} | {v['accuracy_clean']:.3f} |")
    e = res["escalation"]
    L += ["", "## Escalation", "",
          f"Gold: {e['n_escalate_gold']} escalate / {e['n_auto_gold']} auto. "
          f"Cost ratio false-auto:false-escalate = {COST_FALSE_AUTO:.0f}:{COST_FALSE_ESCALATE:.0f}.", "",
          "| policy | acc (w) | esc. precision | esc. recall | false-auto | false-esc | cost | cost (w) |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("predicted_intent", "gold_intent"):
        v = e[k]
        L.append(f"| {k} | {v['accuracy_weighted']:.3f} | {v['escalate_precision']:.2f} | "
                 f"{v['escalate_recall']:.2f} | {v['false_auto']} | {v['false_escalate']} | "
                 f"{v['expected_cost']:.0f} | {v['expected_cost_weighted']:.1f} |")
    L.append(f"| escalate everything | - | - | 1.00 | 0 | {e['escalate_everything']['false_escalate']} "
             f"| {e['escalate_everything']['expected_cost']:.0f} | - |")
    if res.get("replies"):
        L += ["", "## Replies (LLM judge)", "",
              "| tier | n | " + " | ".join(CRITERIA) + " | overall | overall (w) | send-ready |",
              "|---|---|" + "---|" * (len(CRITERIA) + 3)]
        for t, v in res["replies"].items():
            L.append(f"| {t} | {v['n']} | " + " | ".join(f"{v[c]:.2f}" for c in CRITERIA)
                     + f" | {v['overall']:.2f} | {v['overall_weighted']:.2f} | {v['pct_send_ready']:.0f}% |")
    if res.get("judge_agreement"):
        a = res["judge_agreement"]
        L += ["", f"## Judge vs human (n={a['n']})", "", "| criterion | agreement | kappa |", "|---|---|---|"]
        for c, v in a["criteria"].items():
            L.append(f"| {c} | {v['agreement']:.1f}% | {v['kappa']:.3f} |")
        o = a["overall"]
        L.append(f"| overall (1-5) | exact {o['exact']:.1f}%, within-1 {o['within_1']:.1f}% | "
                 f"spearman {o['spearman']:.3f}, bias {o['judge_minus_human']:+.2f} |")
    else:
        L += ["", "## Judge vs human", "", "**Not yet measured.** Run `python judge.py --human --n 40` "
              "then `python judge.py --agreement`. Until then the judge scores above are unvalidated."]
    L += ["", f"_Generated by evaluate.py in {res['seconds']:.0f}s; classifier/generator: "
          f"{res.get('llm', 'none')}; judge: {res.get('judge_llm') or res.get('llm', 'none')}. "
          "Zero calls means every LLM output came from the committed caches under data/._"]
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwritten: {RESULTS / 'results.json'}, {RESULTS / 'RESULTS.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=GOLD_R1)
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample size")
    ap.add_argument("--classifier", default="llm", choices=["majority", "keyword", "llm"],
                    help="classifier that drives escalation (all three are scored on intent)")
    ap.add_argument("--skip-replies", action="store_true")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-model", default=None,
                    help="judge with a different model than the generator (reduces "
                         "self-preference; same backend)")
    args = ap.parse_args()

    t0 = time.time()
    gold = load_gold(args.gold, args.limit)
    print(f"golden set: {len(gold)} rows  strata: "
          + ", ".join(f"{k}={v}" for k, v in Counter(r["stratum"] for r in gold).items()))

    client = None
    need_llm = args.classifier == "llm" or not args.skip_replies
    if need_llm:
        try:
            from llm import Client
            client = Client(backend=args.backend, model=args.model)
            print(f"llm: {client.backend}/{client.model}")
        except SystemExit as e:
            print(f"no LLM backend ({e}); falling back to keyword classifier, replies skipped")
            args.classifier = "keyword" if args.classifier == "llm" else args.classifier
            args.skip_replies = True

    names = ["majority", "keyword"] + (["llm"] if client else [])
    retriever = Retriever.load()
    res = {"n": len(gold), "gold_file": str(args.gold.name), "limit": args.limit,
           "classifier": args.classifier, "llm": str(client) if client else None,
           # Provenance of the labels themselves: empty for a human round.
           "annotators": dict(Counter(r["annotator"] for r in gold if r.get("annotator")))}
    res["intent"], preds_by = eval_intent(gold, names, client)
    res["escalation"] = eval_escalation_all(gold, preds_by[args.classifier], retriever)
    if not args.skip_replies:
        judge_client = client
        if args.judge_model and client:
            from llm import Client
            judge_client = Client(backend=client.backend, model=args.judge_model)
            res["judge_llm"] = str(judge_client)
        res["replies"] = eval_replies(gold, retriever, client, args.tiers.split(","),
                                      judge_client=judge_client)
    try:
        from judge import agreement
        res["judge_agreement"] = agreement()
    except SystemExit:
        res["judge_agreement"] = None
    res["seconds"] = time.time() - t0
    res["llm"] = str(client) if client else None
    write_results(res)


if __name__ == "__main__":
    utf8_console()
    main()
