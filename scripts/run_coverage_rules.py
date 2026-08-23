"""
Manual verification of src/rules/coverage_rules.py against real uploaded
data — first real-data test of the deterministic rule engine (previously
only mocked in tests/). No TestCase nodes exist yet (sample_testcases.json
not authored), so expect 0% coverage everywhere — that's the correct,
expected result right now, not a bug. This script's job is to confirm the
QUERIES are logically sound, not that coverage is good.

Run: python scripts/run_coverage_rules.py
"""
from src.graph.neo4j_client import Neo4jClient
from src.rules.coverage_rules import (
    get_coverage_summary,
    find_missing_coverage,
    find_low_coverage_regulations,
    find_high_risk_gaps,
)

with Neo4jClient() as client:
    print("=" * 70)
    print("COVERAGE SUMMARY (per regulation)")
    print("=" * 70)
    summary = get_coverage_summary(client)
    for row in summary:
        print(f"  {row['doc_id']:<30} {row['covered_clauses']}/{row['total_clauses']} "
              f"({row['coverage_pct']:.1f}%)")

    print("\n" + "=" * 70)
    print("LOW COVERAGE REGULATIONS (< 80% threshold)")
    print("=" * 70)
    low = find_low_coverage_regulations(client)
    print(f"  {len(low)} regulation(s) below threshold (expected: all, until TestCases exist)")

    print("\n" + "=" * 70)
    print("HIGH-RISK MISSING COVERAGE (first 10)")
    print("=" * 70)
    gaps = find_high_risk_gaps(client)
    print(f"  Total high-risk gaps: {len(gaps)}")
    for row in gaps[:10]:
        print(f"  [{row['doc_id']}] {row['clause_id']}")
        print(f"    {row['text'][:100]}...")
