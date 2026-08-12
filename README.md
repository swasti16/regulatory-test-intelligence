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

## Architecture

```
PDF (regulation doc)
      ↓
Docling — text/structure extraction
      ↓
LLM (Ollama, local) — clause identification + risk classification
      ↓
Neo4j — traceability graph
      ↓
Deterministic Cypher rules — coverage gap detection
```

### Graph Schema

```
(Regulation) -[:HAS_CLAUSE]-> (Clause {risk_level}) -[:COVERED_BY]-> (TestCase)
```

### Why deterministic rules, not LLM-guessed gaps

Compliance gap detection needs to be auditable and reproducible. An LLM
"guessing" which clauses lack coverage introduces hallucination risk in a
domain where false negatives (missed gaps) have real regulatory
consequences. Rules here are plain Cypher queries — traceable, testable,
version-controlled.

**MVP Rules (Phase 1):**
1. **Missing Coverage** — clauses with zero linked test cases
2. **Low Coverage Threshold** — regulations below 80% clause coverage
3. **High-Risk Prioritization** — missing-coverage clauses filtered by `risk_level: high`

> Framing note: this MVP applies fixed, hand-written rules — not
> auto-generated test scenarios or continuous recommendation. Roadmap
> items (see below) are explicitly future work, not current capability.

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
│   └── settings.py            # Central config — Neo4j, Ollama, thresholds
├── src/
│   ├── ingestion/              # Docling PDF -> structured text
│   ├── extraction/             # LLM clause extraction
│   ├── graph/                  # Neo4j driver + graph writes
│   ├── rules/                  # Deterministic Cypher coverage rules
│   └── orchestration/          # LangGraph pipeline wiring
├── tests/
│   ├── graph/
│   ├── ingestion/
│   └── integration/
├── data/sample_regulations/    # Sample PDFs for dev/testing
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

- [x] Neo4j schema design + manual rule validation (Neo4j Browser)
- [ ] Neo4j Python driver connection
- [ ] Docling PDF ingestion
- [ ] LLM clause extraction pipeline
- [ ] Graph write pipeline (extracted clauses -> Neo4j)
- [ ] Deterministic rule engine (Python wrapper over Cypher queries)
- [ ] LangGraph orchestration
- [ ] LangSmith tracing
- [ ] MCP server over Neo4j (Phase 2 — not in hackathon MVP scope)

## Author

**Swasti Shrivastava** — [@swasti16](https://github.com/swasti16)
Built for Coforge TechCon 2026 Hackathon (Team: Syntax Terror)
