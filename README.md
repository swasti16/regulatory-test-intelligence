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
human sign-off on high-risk clause links, and LLM-assisted link *suggestion*
(never auto-linking), are Post-MVP roadmap items (see below) — deliberately
scoped out of the MVP because an ungoverned LLM linker would contradict this
project's own core principle (LLM proposes, deterministic/human logic decides),
and a wrong auto-link is strictly worse than a visible gap per the failure-mode
table above.

## Architecture

```mermaid
flowchart TD
    A[PDF Regulation Doc] --> B[Docling text/structure extraction]
    B --> C[LLM Ollama - clause ID and risk classification]
    C --> D[Deterministic Post-Processing]
    D --> E[extracted_clauses JSON - HUMAN REVIEW CHECKPOINT]
    E --> F[(Neo4j Graph)]
    G[sample_testcases.json] --> H[upload_test_cases.py]
    H --> F
    F --> I[Deterministic Cypher Rules]
    I --> J[Coverage Gap Report]
```

### Graph Schema

```
(Regulation) -[:HAS_CLAUSE]-> (Clause {risk_level}) -[:COVERED_BY]-> (TestCase)
```

### Human Review Checkpoints

Two deliberate checkpoints separate slow/non-deterministic steps from fast/
deterministic ones, and keep humans in the loop on trust-critical decisions:

- **Extraction → Graph**: **`scripts/extract_to_json.py`** runs LLM extraction
  + all deterministic post-processing, writes results to
  `data/extracted_clauses/{doc_id}.json`. No Neo4j writes. Every clause
  (including dropped ones) is retained in the JSON with a `status` field:
  `included`, `dropped_ungrounded`, or `dropped_illustrative`.
  **`scripts/upload_to_neo4j.py`** reads the reviewed JSON, uploads
  only `status == "included"` clauses.
- **Test-case linking**: **`data/sample_testcases.json`** holds test-case-to-
  clause links, each with its own `status` field (`confirmed` today; a future
  LLM-assisted matcher would add `suggested` entries for human review).
  **`scripts/upload_test_cases.py`** writes only `status == "confirmed"`
  links to Neo4j, and verifies each referenced clause actually exists before
  linking — a typo'd `chapter_title`/`clause_num` fails loud, not silent.

This means a human can inspect exactly what was extracted, what was
filtered out and why, and which test-case links are trusted — before
anything touches the graph. It also decouples the slow/non-deterministic
steps (LLM inference, ~15-80 min per chapter on CPU) from the fast/
deterministic ones (Neo4j write), so a graph-write failure never forces
re-running extraction.

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
  instruction before this override was added. Known tradeoff: this also
  flags purely procedural "shall"-containing boilerplate (e.g. short-title/
  commencement clauses) as high-risk alongside substantively risky clauses
  — a precision/recall tradeoff favoring recall (never silently miss a real
  hard-obligation clause) at the cost of some low-value high-risk noise.
- **Illustrative-example filter** (`_filter_illustrative()`) — drops
  "Illustration:"/"Example:" clauses, which often contain signal words
  describing a scenario (not a rule) and would otherwise be
  false-positively force-labeled high.

**MVP Rules (Phase 1):**
1. **Missing Coverage** — clauses with zero linked test cases
2. **Low Coverage Threshold** — regulations below 80% clause coverage
3. **High-Risk Prioritization** — missing-coverage clauses filtered by `risk_level: high`


## LangGraph Orchestration

`src/orchestration/` wraps the extraction pipeline (load PDF → split into
sections → LLM extract → write JSON) as an explicit **state graph** instead
of a linear script, giving conditional retry logic a real cycle to run in.

**Why LangGraph over a plain LangChain chain:** grounding-failure retry
needs a *loop* — extract a section, check if anything survived, and if not,
split the section and try again. LangChain's chains are linear (no cycles);
LangGraph models this as a graph with conditional edges, so the retry loop
is explicit and inspectable rather than hand-rolled control flow.

**Flow:**
```mermaid
stateDiagram-v2
    [*] --> load
    load --> split
    split --> extract
    extract --> advance
    extract --> split_retry
    split_retry --> advance
    split_retry --> mark_failed
    mark_failed --> advance
    advance --> extract
    advance --> write
    write --> [*]
```

**Retry strategy — split-half, not temperature bump:** on zero-clauses-
survived, the section is split at a *sentence* boundary (never mid-clause)
with a 2-sentence overlap between halves, and each half is re-extracted
independently. This targets the actual failure mode already documented in
`docling_loader.py` — long sections cause "lost in the middle" attention
degradation — rather than hoping a different sampling temperature helps.
Overlap-caused duplicate `clause_num`s are deduped with the same `#1`/`#2`
suffix pattern used in `post_process_extraction.py`.

**Resilience:** a per-section extraction failure (e.g. Ollama timeout) is
caught and treated as "0 survived," routing to retry/mark-failed rather
than crashing the whole PDF — mirrors `extract_to_json.py`'s per-section
try/except so a single bad section can't lose an entire document's output.

**Scope (deliberate):** single-PDF, in-process, no subprocess isolation —
unlike `extract_to_json.py`'s per-PDF subprocess isolation for OOM safety
on batch runs. `scripts/run_langgraph_pipeline.py` supports both a single
PDF path or, with no argument, loops every PDF in `Settings.RBI_PDF_DIR`
sequentially in one process. Multi-PDF subprocess isolation, Neo4j writes,
and LangSmith-traced production hardening are deferred post-hackathon-demo.

**Observability:** LangSmith tracing (env-var activated, no code changes
to `graph.py` needed) gives a visual trace tree per run — every node
execution, every LLM call's prompt/response, and router decisions,
viewable at smith.langchain.com.

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
│   ├── post_process_extraction.py # Re-extract failures, dedupe, validate readiness
│   ├── upload_to_neo4j.py        # Reviewed JSON -> Neo4j (status=="included" only)
│   ├── upload_test_cases.py      # sample_testcases.json -> Neo4j (status=="confirmed" only)
│   ├── clear_neo4j.py            # Wipes all nodes/relationships (dev reset)
│   ├── reextract_sections.py     # Re-run extraction on specific failed sections
│   ├── analyse_drops.py          # Diagnoses dropped_ungrounded clauses
│   └── run_coverage_rules.py     # Manual verification of coverage rules against live data
├── tests/
│   ├── graph/
│   ├── ingestion/
│   ├── extraction/
│   └── integration/
├── data/
│   ├── sample_regulations/       # Sample PDFs for dev/testing
│   ├── RBI_regulations/          # Real RBI PDFs (gitignored)
│   ├── extracted_clauses/        # Extraction output — human review checkpoint (gitignored)
│   └── sample_testcases.json     # Human-curated test-case-to-clause seed links
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

## Known Limitations

- **Paraphrase vs. verbatim list-item extraction**: the grounding check
  (`_is_grounded_in_source()`) uses fuzzy fragment matching (≥0.6 weighted
  ratio), not exact substring matching — so a clause the LLM lightly
  rephrases (e.g. reordering a list item's wording) can still pass
  grounding even though it isn't a verbatim quote. This is intentional
  (exact-substring would reject valid paraphrases and over-drop real
  clauses), but it does mean "grounded" is not a guarantee of "verbatim."
- **Inferred page boundaries**: `page_start`/`page_end` on a clause are
  derived from the nearest `[p.N]` marker in the reconstructed text
  stream, not from the clause's own precise bounding box — a clause that
  spans a page break may report a slightly imprecise page range.
- **Shared-stem list-item grounding limitation**: when several list items
  in the source share a long common prefix (e.g. repeated "The bank
  shall..." across sub-bullets), the fuzzy grounding check can occasionally
  match a clause against the wrong sibling item rather than its true
  source fragment, since both score similarly high.
  - **`clause_id` is not guaranteed stable across re-extraction runs** — `clause_num`
  is currently LLM-assigned free text (e.g. `"11(1)"`, `"B.2"`). Even at
  `temperature: 0.0`, Ollama sampling isn't bit-for-bit deterministic across
  runs, so the LLM's numbering/segmentation can drift between runs of the
  same PDF. Since `clause_id = {doc_id}_{chapter_title}_{clause_num}` is the
  Neo4j MERGE key, drift causes: (1) re-extraction creates orphaned duplicate
  nodes instead of updating existing ones, breaking idempotency; (2)
  hardcoded clause references in `data/sample_testcases.json` (11 links)
  can silently point to a clause that no longer exists under that ID.

  **Planned fix (deferred to post-evaluation, Round 3+):** anchor `clause_id`
  to the clause's matched source-text position rather than the LLM's own
  label — the grounding check (`_is_grounded_in_source()` /
  `_fragment_grounded()`) already computes a fuzzy-matched window against
  fixed source tokens; capturing that window's start-index and using it to
  assign a positional ID (e.g. `pos_0001`, `pos_0002`, ordered by source
  position) would make IDs deterministic since they're anchored to
  immutable source text, not LLM phrasing. Original LLM label would be kept
  as a separate `clause_label` field for human readability.

  Estimated effort: ~2.5-3.5 hours, most of which is remapping
  `sample_testcases.json`'s 11 hardcoded clause references and re-verifying
  Neo4j/test-case links end-to-end, not the core anchor-ID logic itself.
  Deferred to avoid destabilizing working TestCase links immediately before
  the hackathon demo; will prioritize after the 3rd evaluation round once
  real-world drift frequency is better understood from actual eval runs.


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
- [x] Model benchmarking harness — speed + rubric-adherence comparison across candidate models; `llama3.2:3b` confirmed as production model
- [x] Full pipeline run across all 5 RBI regulation PDFs — extracted, uploaded to Neo4j Aura
- [x] Grounding normalization fix (`_normalize_text_clean`) — PDF source renders "his / her" as 3 tokens vs LLM output "his/her" as 1 token, breaking the sliding-window fuzzy match on otherwise-correctly-extracted clauses; fixed by collapsing slash-spacing symmetrically in both fragment and source normalization
- [x] Deterministic rule engine (`src/rules/coverage_rules.py`) — 3 MVP Cypher rules verified against real uploaded Neo4j data
- [x] Post-processing pipeline (`scripts/post_process_extraction.py`) — re-extracts failed sections, fixes leaked clause_nums, dedupes collisions, validates Neo4j-upload readiness (FAIL/WARN checklist)
- [x] TestCase seed data (`data/sample_testcases.json`) — 12 hand-curated test cases across all 5 RBI PDFs, linking to 11 distinct clauses. Schema deliberately shaped to mirror a future Jira/TestRail import adapter's output (`source`, `covers[]` with per-link `status`), not a one-off fixture. Demonstrates both one-to-many (one test covering 2 clauses) and many-to-one (2 tests covering the same clause) mappings.
- [x] TestCase upload script (`scripts/upload_test_cases.py`) — resolves `(doc_id, chapter_title, clause_num)` → composite `clause_id` via the existing single-source-of-truth `clause_id()` function; only writes links with `status: "confirmed"`; verifies clause existence before linking (fails loud on typos, not silent)
- [x] End-to-end coverage verification against real Neo4j Aura data: 12 links written, 0 missing-clause errors, coverage % now non-zero across all 5 regulations (0.4%–1.5%), 988 high-risk gaps still correctly surfaced by `find_high_risk_gaps()` — proves the full pipeline (extraction → graph → coverage detection) works end-to-end on real data, not just mocks

**In progress:**
- [ ] Duplicate clause fix (Section J duplication bug — same requirement extracted twice with overlapping text spans)
- [ ] LangGraph orchestration (`src/orchestration/`) — replace linear script with explicit state graph, conditional retry/branching
- [ ] LangSmith tracing — per-node observability

**Not started:**
- [ ] Human-in-the-loop review queue for clause-to-test-case linking, prioritized by `risk_level`
- [ ] LLM-assisted test-case-to-clause link suggestion (Post-MVP, explicitly deferred) — an ungoverned LLM linker would contradict the project's core "LLM proposes, deterministic/human decides" principle, since wrong links are strictly worse than visible gaps (see failure-mode table above). Schema is forward-compatible: a future matcher would append entries with `status: "suggested"`; only human-promoted `"confirmed"` links are ever written to the graph. Deferred in favor of finishing LangGraph/LangSmith given a 3-day hackathon timeline.
- [ ] **Deterministic `clause_id` scheme** — anchor `clause_id` to source-text
  match position (from the existing grounding check) instead of LLM-assigned
  `clause_num`, to guarantee idempotent Neo4j MERGE and stable
  `sample_testcases.json` links across re-extraction runs.
- [ ] Real-time compliance coverage dashboard for QA leads
- [ ] MCP server over Neo4j (Phase 2)

## Author

**Swasti Shrivastava** — [@swasti16](https://github.com/swasti16)
Built for Coforge TechCon 2026 Hackathon (Team: Syntax Terror)
