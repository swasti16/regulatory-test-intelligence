"""
Post-hoc quality scoring over benchmark_results.json — no model reruns
needed, works off saved output from scripts/benchmark_models.py.

Scoring heuristic: RISK_RUBRIC (clause_extractor.py) explicitly says
"high" applies to clauses with hard deadlines/penalties/mandatory
requirements, citing "shall", "must", "penalty of" as signal language.
This is a cheap, imperfect proxy — not authoritative, since rubric intent
has nuance beyond keyword matching (e.g. "shall" appears in low-risk
definitional sentences too) — but it turns "feels more sensible" into a
comparable number across models rather than eyeballing a few printed
clauses.
"""
import json

SIGNAL_WORDS = ["shall", "must", "penalty of"]

with open("benchmark_results.json", encoding="utf-8") as f:
    results = json.load(f)

print(f"{'Model':<18}{'Chapter':<22}{'Clauses':<10}{'Signal-word':<14}{'Mislabeled':<12}{'Violation %'}")

for r in results:
    clauses = r["clauses"]
    signal_clauses = [c for c in clauses if any(w in c["text"].lower() for w in SIGNAL_WORDS)]
    mislabeled = [c for c in signal_clauses if c["risk_level"] != "high"]
    violation_pct = (len(mislabeled) / len(signal_clauses) * 100) if signal_clauses else 0.0

    print(f"{r['model']:<18}{r['chapter']:<22}{len(clauses):<10}{len(signal_clauses):<14}{len(mislabeled):<12}{violation_pct:.0f}%")

    if mislabeled:
        for c in mislabeled:
            print(f"    MISLABELED [{c['risk_level']}] {c['clause_num']}: {c['text'][:90]}...")
