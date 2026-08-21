# Regulatory Test Intelligence

AI-assisted pipeline that reads banking regulatory compliance PDFs, extracts
clauses via LLM, and builds a traceability graph linking regulations to
test coverage — surfacing compliance gaps via deterministic rule checks
(not AI-guessed).

## Problem

Compliance teams manually cross-reference regulatory documents against QA
test suites to confirm coverage. This is slow, error-prone, and doesn't
scale as regulations change. This project automates clause extraction and
coverage traceability, while keeping the actual gap-detection logic
deterministic and auditable — critical for a regulated domain like banking.

## Why This Matters — The Cost of the Gap

Banks face continuous regulatory change. Each RBI amendment triggers regression
and compliance testing across core banking, payments, and reporting modules —
today, done by manually re-reading the regulation and manually checking which
test cases (if any) cover it. There is no automated link between a regulatory
clause and the test evidence that proves it's enforced.

**What happens when that manual link breaks:**

| Failure Mode | Consequence |
|---|---|
| Clause never mapped to any test | Gap stays invisible — no one is even looking for it |
| Clause mapped incorrectly ("looks covered") | **Worse than a visible gap** — false confidence, the breach surfaces first in production, not in QA |
| Regulation updates, mapping isn't refreshed | Stale coverage — tests pass against an old rule that no longer matches the current directive |
| Detection happens late (weeks, per manual cycle) | Non-compliant code ships in the release window before the gap is caught |

**What that costs, concretely** — drawn directly from the sample regulation this
project processes (RBI Commercial Banks Credit/Debit Card Directions, 2025):

- **Direct financial penalty**: failure to close a card account within 7 working
  days of a valid request carries a **₹500/day penalty**, payable to the customer,
  for every day of delay (Ch. II-E). An untested code path here compounds daily.
- **Punitive multiplier**: an unsolicited card issued and billed without consent
  requires the bank to reverse the charge **and** pay a penalty of **twice the
  reversed amount**, on top of Ombudsman-determined compensation for the
  customer's time, harassment, and mental anguish (Ch. II-C).
- **Regulatory escalation**: every unresolved failure category has an explicit
  RBI Ombudsman path (Ch. VI-D) — meaning gaps don't just risk a fine, they risk
  a formal regulatory finding against the bank, with reputational and (in
  repeat/severe cases) licensing consequences.
- **Time cost**: this single 35-page directions document has ~8 chapters and
  100+ individually testable obligations. Manually diffing that against a test
  suite, per amendment, is a multi-day task for a compliance/QA analyst — RBI
  issues Master Directions and amendments multiple times a year (this document
  itself explicitly repeals and replaces a prior 2025 circular).

**How this project addresses each failure mode:**

| Gap | This Project's Answer |
|---|---|
| Invisible coverage gaps | Deterministic Cypher rules surface every clause with zero linked test cases — visible before release, not after a breach |
| False "covered" confidence | Grounding check (`_is_grounded_in_source()`) — every extracted clause must be a verbatim substring of the source regulation, so extraction can't silently drift or hallucinate coverage that isn't real |
| Stale mappings after amendments | Idempotent, MERGE-based graph writes — re-running extraction on an updated PDF safely refreshes clauses without duplicating or losing existing test-case links |
| Slow manual detection | Extraction + gap-surfacing runs in minutes per document vs. days of manual cross-referencing; analyst time shifts from *searching* for gaps to *validating* flagged ones |
| Trusting AI blindly on a compliance-critical decision | The LLM only extracts and classifies — it never decides what counts as a "gap." That decision is a fixed, auditable Cypher rule, not an LLM judgment call |

**What's explicitly NOT automated yet (by design, not oversight):**
Clause-to-test-case linking in this MVP is human-authored seed data — a
compliance/QA lead still decides which test case satisfies which clause.
The system's job is to make gaps *visible and current*, not to remove the
human decision of "is this test actually sufficient." A review queue for
human sign-off on high-risk clause links is a Post-MVP roadmap item (see below).

## Architecture

```
PDF (regulation doc)
      ↓
Docling — text/structure extraction, per-section split
      ↓
LLM (Ollama, local) — clause identification + risk classification
      ↓
Deterministic post-processing — grounding check, illustrative-example
filter, risk-rubric override (see "Why deterministic rules" below)
      ↓
data/extracted_clauses/{doc_id}.json — human review checkpoint
      ↓
Neo4j — traceability graph (upload only status=="included" clauses)
      ↓
Deterministic Cypher rules — coverage gap detection
```

### Graph Schema

```
(Regulation) -[:HAS_CLAUSE]-> (Clause {risk_level}) -[:COVERED_BY]-> (TestCase)
```

### Human Review Checkpoint

Extraction and graph-write are deliberately split into two scripts, not
one pipeline:

- **`scripts/extract_to_json.py`** — runs LLM extraction + all
  deterministic post-processing, writes results to
  `data/extracted_clauses/{doc_id}.json`. No Neo4j writes. Every clause
  (including dropped ones) is retained in the JSON with a `status` field:
  `included`, `dropped_ungrounded`, or `dropped_illustrative`.
- **`scripts/upload_to_neo4j.py`** — reads the reviewed JSON, uploads
  only `status == "included"` clauses.

This means a human can inspect exactly what was extracted and what was
filtered out — and why — before anything touches the graph. It also
decouples the slow/non-deterministic step (LLM inference, ~15-80 min per
chapter on CPU) from the fast/deterministic step (Neo4j write), so a
graph-write failure never forces re-running extraction.

### Why deterministic rules, not LLM-guessed gaps

Compliance gap detection needs to be auditable and reproducible. An LLM
"guessing" which clauses lack coverage introduces hallucination risk in a
domain where false negatives (missed gaps) have real regulatory
consequences. Rules here are plain Cypher queries — traceable, testable,
version-controlled.

Two more deterministic layers sit between LLM extraction and the graph:
- **Grounding check** (`_is_grounded_in_source()`) — every extracted
  clause must fuzzy-match a fragment of the actual source text, rejecting
  fabricated/hallucinated clauses before they're even written to JSON.
- **Risk-rubric override** (`_enforce_risk_rubric()`) — any clause
  containing a hard signal word ("shall", "shall not", "must") is
  force-labeled `high`, regardless of what the LLM assigned. Measured
  ~43% LLM self-application failure rate on the rubric's own few-shot
  instruction before this override was added.
- **Illustrative-example filter** (`_filter_illustrative()`) — drops
  "Illustration:"/"Example:" clauses, which often contain signal words
  describing a scenario (not a rule) and would otherwise be
  false-positively force-labeled high.

**MVP Rules (Phase 1):**
1. **Missing Coverage** — clauses with zero linked test cases
2. **Low Coverage Threshold** — regulations below 80% clause coverage
3. **High-Risk Prioritization** — missing-coverage clauses filtered by `risk_level: high`

## Tech Stack

| Component | Technology |
|---|---|
| PDF Ingestion | Docling |
| LLM | Ollama (local, teammate-hosted) |
| Graph DB | Neo4j (Aura Free) |
| Orchestration | LangGraph |
| Observability | LangSmith |
| Testing | Pytest |

## Project Structure

```
regulatory-test-intelligence/
├── config/
│   └── settings.py              # Central config — Neo4j, Ollama, thresholds
├── src/
│   ├── ingestion/                # Docling PDF -> chapter/section chunks
│   ├── extraction/                # LLM clause extraction + deterministic filters
│   ├── graph/                    # Neo4j driver + graph writes
│   ├── rules/                    # Deterministic Cypher coverage rules
│   └── orchestration/            # LangGraph pipeline wiring (not started)
├── scripts/
│   ├── extract_to_json.py        # PDF -> data/extracted_clauses/{doc_id}.json
│   ├── upload_to_neo4j.py        # Reviewed JSON -> Neo4j (status=="included" only)
│   ├── clear_neo4j.py            # Wipes all nodes/relationships (dev reset)
│   ├── benchmark_model.py        # Speed + rubric-adherence model comparison
│   └── analyse_benchmark.py      # Post-hoc scoring over benchmark_results.json
├── tests/
│   ├── graph/
│   ├── ingestion/
│   ├── extraction/
│   └── integration/
├── data/
│   ├── sample_regulations/       # Sample PDFs for dev/testing
│   ├── RBI_regulations/          # Real RBI PDFs (gitignored)
│   └── extracted_clauses/        # Extraction output — human review checkpoint (gitignored)
└── docs/
```

## Setup

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Configure
cp .env.example .env
# Add your Neo4j Aura credentials and teammate's Ollama endpoint
```

## Roadmap

**Built & tested:**
- [x] Neo4j schema design + manual rule validation (Neo4j Browser)
- [x] Neo4j Python driver connection (`Neo4jClient` — mocked unit tests + real Aura idempotency tests)
- [x] Docling PDF ingestion (chapter-level chunking, page-provenance `[p.N]` markers, bbox-sort reading-order fix)
- [x] Sub-chapter section splitting (`split_chapter_into_sections()`) — fixes "lost in the middle" attention failures on long chapters
- [x] LLM clause extraction pipeline (Ollama, rubric + few-shot prompting, JSON-constrained output, `<CHAPTER_TEXT>` boundary tags to prevent prompt-injection-as-clause bugs)
- [x] Grounding check (`_is_grounded_in_source()`) — fuzzy fragment-matching against source text, rejects fabricated clauses
- [x] Deterministic risk-rubric override (`_enforce_risk_rubric()`) — signal-word force-labeling, 0% violation rate on latest run (down from 42.7%)
- [x] Illustrative-example filter (`_filter_illustrative()`) — drops "Illustration:"/"Example:" clauses from enforceable output
- [x] Extraction/upload pipeline split — `extract_to_json.py` (LLM + filters, no DB writes) and `upload_to_neo4j.py` (reviewed JSON -> graph), enabling a human review checkpoint between the two
- [x] Graph write pipeline (`graph_writer.py` — idempotent MERGE for Regulation/Clause/TestCase nodes and relationships)
- [x] Model benchmarking harness (`benchmark_model.py`, `analyse_benchmark.py`) — speed + rubric-adherence comparison across candidate models; `llama3.2:3b` confirmed as production model

**In progress:**
- [ ] Deterministic rule engine (`src/rules/coverage_rules.py`) — module has the 3 MVP Cypher rules written, needs test coverage + a demo run against uploaded data
- [ ] Duplicate clause fix (Section J duplication bug — same requirement extracted twice with overlapping text spans)

**Not started:**
- [ ] Full pipeline run across all 5 RBI regulation PDFs (only Chapter II/III of 1 PDF processed so far)
- [ ] TestCase seed data (`data/sample_testcases.json`) — blocked until extraction output is stable across a full document
- [ ] LangGraph orchestration (`src/orchestration/`) — replace linear script with explicit state graph, conditional retry/branching
- [ ] LangSmith tracing — per-node observability
- [ ] Human-in-the-loop review queue for clause-to-test-case linking, prioritized by `risk_level`
- [ ] Auto-suggested test case generation from extracted clauses (Post-MVP — a recommendation surface only; QA lead always approves before anything enters the real test suite)
- [ ] Real-time compliance coverage dashboard for QA leads
- [ ] MCP server over Neo4j (Phase 2)

## Author

**Swasti Shrivastava** — [@swasti16](https://github.com/swasti16)
Built for Coforge TechCon 2026 Hackathon (Team: Syntax Terror)
