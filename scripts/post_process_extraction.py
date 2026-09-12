"""
Post-processing pass — runs AFTER extract_to_json.py's batch completes,
BEFORE upload_to_neo4j.py. Mutates each data/extracted_clauses/{doc_id}.json
in place (writes a .bak backup first) and prints a per-file readiness
report for the human review checkpoint.

Order matters and differs slightly from a naive read of the 4 requested
steps — reasoning:
  1. Re-extract failed sections FIRST (only step that adds new clauses;
     everything after must operate on the final, complete clause list).
  2. Fix leaked clause_nums SECOND (leaked ones are globally unique by
     list index once fixed, so this can never itself cause a collision).
  3. Dedupe clause_nums THIRD — safety net over the now-final list. Given
     reextract_sections.py fully strips a section's old clauses before
     re-adding new ones, collisions between kept + new clauses should be
     structurally impossible here, but this is cheap insurance, not
     wasted work.
  4. Analyse drops LAST, read-only/diagnostic only — same as the existing
     analyse_drops.py behavior. This script does NOT auto-apply grounding
     flips (that was patch_grounding_fix.py's job, and it's now stale/
     hardcoded to old clause_nums from a prior prompt version — do not
     reuse it going forward, see cleanup notes).
  5. Validate LAST, and persist validation_issues into the JSON itself —
     this is what upload_to_neo4j.py's gate checks. A single _save() call
     after this step captures both post-processing fixes and the
     validation verdict in one write.

Run:
    python scripts/post_process_extraction.py                # all files
    python scripts/post_process_extraction.py RBI_KYC          # one doc_id
"""
import json
import os
import sys
import glob
import shutil
from collections import Counter

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from scripts.reextract_sections import reextract
from scripts.analyse_drops import analyse_file

INPUT_DIR = "data/extracted_clauses"
_LEAKED_CLAUSE_NUM_MAX_LEN = 30
_FAILED_SECTION_SUFFIX = " (0 survived — needs retry)"


def _load(doc_id: str) -> dict:
    path = os.path.join(INPUT_DIR, f"{doc_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(doc_id: str, data: dict) -> None:
    path = os.path.join(INPUT_DIR, f"{doc_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _backup(doc_id: str) -> None:
    src = os.path.join(INPUT_DIR, f"{doc_id}.json")
    dst = os.path.join(INPUT_DIR, f"{doc_id}.json.bak")
    shutil.copy2(src, dst)


def _fix_leaked_clause_nums(clauses: list) -> int:
    """
    clause_num > threshold chars means the LLM echoed clause TEXT into
    the numbering field instead of a real label. Reassigns a placeholder
    that is unique BY CONSTRUCTION (list index is monotonic and unique
    file-wide) — no follow-up collision possible from this step alone.
    """
    fixed = 0
    for idx, c in enumerate(clauses):
        if len(c["clause_num"]) > _LEAKED_CLAUSE_NUM_MAX_LEN:
            c["clause_num"] = f"leaked_fixed_{idx}"
            fixed += 1
    return fixed


def _dedupe_clause_nums(clauses: list) -> int:
    """
    Same algorithm as extract_to_json.py's own disambiguation — mirrored
    here (not imported) because extract_to_json's version is inline in a
    larger function, not a standalone reusable unit. Idempotent: running
    this twice on an already-unique list makes zero further changes.
    """
    key_counts = Counter((c["chapter_title"], c["clause_num"]) for c in clauses)
    seen = Counter()
    changed = 0
    for c in clauses:
        key = (c["chapter_title"], c["clause_num"])
        if key_counts[key] > 1:
            seen[key] += 1
            new_num = f"{c['clause_num']}#{seen[key]}"
            if new_num != c["clause_num"]:
                changed += 1
            c["clause_num"] = new_num
    return changed


def _recompute_summary(data: dict) -> None:
    clauses = data["clauses"]

    def _count(status):
        return sum(1 for c in clauses if c["status"] == status)

    data["summary"]["total_extracted"] = len(clauses)
    data["summary"]["included"] = _count("included")
    data["summary"]["dropped_ungrounded"] = _count("dropped_ungrounded")
    data["summary"]["dropped_illustrative"] = _count("dropped_illustrative")
    data["summary"]["dropped_invalid_risk"] = _count("dropped_invalid_risk")


def _reextract_failed_sections(doc_id: str, data: dict) -> bool:
    """Returns True if reextract() ran (and re-saved the file), False if
    there was nothing to do — caller must reload data after True."""
    failed = data.get("summary", {}).get("failed_sections", [])
    targets = [
        f.replace(_FAILED_SECTION_SUFFIX, "")
        for f in failed
        if f.endswith(_FAILED_SECTION_SUFFIX)
    ]
    if not targets:
        return False
    print(f"  Re-extracting {len(targets)} failed section(s)...")
    reextract(doc_id, targets)  # re-saves the file itself
    return True


VALID_RISK_LEVELS = {"high", "medium", "low"}
_NOISE_TEXT_LITERALS = {"penalty of", "within x days", "shall", "must"}


def _validate_for_upload(doc_id: str, data: dict) -> list:
    """Read-only checks. Returns list of issue strings — empty = ready
    to upload. Warnings (soft) are prefixed 'WARN:', hard failures are
    prefixed 'FAIL:'."""
    issues = []

    if data.get("doc_id") != doc_id:
        issues.append(f"FAIL: doc_id field {data.get('doc_id')!r} != filename {doc_id!r}")

    clauses = data.get("clauses", [])
    s = data.get("summary", {})
    actual_total = len(clauses)
    if s.get("total_extracted") != actual_total:
        issues.append(f"FAIL: summary.total_extracted={s.get('total_extracted')} != actual {actual_total}")

    included = [c for c in clauses if c.get("status") == "included"]

    seen_ids = set()
    for c in included:
        cid = (data["doc_id"], c.get("chapter_title"), c.get("clause_num"))
        if cid in seen_ids:
            issues.append(f"FAIL: duplicate clause_id for Neo4j MERGE key: {cid}")
        seen_ids.add(cid)

        if not c.get("text", "").strip():
            issues.append(f"FAIL: included clause with empty text — clause_num={c.get('clause_num')}")
        if c.get("risk_level") not in VALID_RISK_LEVELS:
            issues.append(f"FAIL: invalid risk_level {c.get('risk_level')!r} — clause_num={c.get('clause_num')}")
        if len(c.get("clause_num", "")) > _LEAKED_CLAUSE_NUM_MAX_LEN:
            issues.append(f"FAIL: clause_num still leaked (>{_LEAKED_CLAUSE_NUM_MAX_LEN} chars) — {c['clause_num'][:50]}...")
        norm_text = c.get("text", "").strip().lower()
        if norm_text in _NOISE_TEXT_LITERALS or len(norm_text) < 15:
            issues.append(f"WARN: suspiciously short/noise-like included clause text: {c.get('text')!r}")
        if c.get("truncated"):
            issues.append(
                f"WARN: truncated clause (Ollama output cut off, may be incomplete) — "
                f"clause_num={c.get('clause_num')}"
            )

    null_pages = sum(1 for c in included if c.get("page_start") is None)
    if null_pages:
        issues.append(f"WARN: {null_pages}/{len(included)} included clause(s) have page_start=None")

    if actual_total < 20:
        issues.append(f"WARN: only {actual_total} total clauses extracted — unusually low, check for a botched/partial run")

    remaining_failed = [f for f in s.get("failed_sections", []) if f.endswith(_FAILED_SECTION_SUFFIX)]
    if remaining_failed:
        issues.append(f"WARN: {len(remaining_failed)} section(s) still 0-survived after reextraction attempt: {remaining_failed}")

    return issues


def process_one(doc_id: str) -> None:
    print(f"\n{'='*70}\n{doc_id}\n{'='*70}")
    _backup(doc_id)

    data = _load(doc_id)
    if _reextract_failed_sections(doc_id, data):
        data = _load(doc_id)  # reextract() rewrote the file — reload

    clauses = data["clauses"]
    n_leaked_fixed = _fix_leaked_clause_nums(clauses)
    n_deduped = _dedupe_clause_nums(clauses)
    _recompute_summary(data)

    print(f"  Leaked clause_nums fixed: {n_leaked_fixed}")
    print(f"  Clause_num collisions deduped: {n_deduped}")
    print(f"  Summary: {data['summary']}")

    print("  --- Grounding drop analysis (diagnostic only, no auto-fix) ---")
    try:
        analyse_file(os.path.join(INPUT_DIR, f"{doc_id}.json"))
    except Exception as e:
        print(f"  (analyse_drops failed: {e})")

    print("  --- Neo4j-upload readiness ---")
    issues = _validate_for_upload(doc_id, data)
    # Persisted into the JSON itself (not just printed) — this is the field
    # upload_to_neo4j.py's _blocking_failures() reads to gate the upload.
    # Without this write, the validation gate has nothing to check for a
    # script-pipeline (non-LangGraph) run and silently no-ops.
    data["validation_issues"] = issues
    _save(doc_id, data)

    if not issues:
        print("  READY — no issues found.")
    else:
        for issue in issues:
            print(f"  {issue}")
        if any(i.startswith("FAIL:") for i in issues):
            print("  NOT READY — resolve FAIL items before uploading.")
        else:
            print("  READY WITH WARNINGS — review WARN items before uploading.")


def main():
    if len(sys.argv) > 1:
        doc_ids = [sys.argv[1]]
    else:
        doc_ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(INPUT_DIR, "*.json"))
        )
    for doc_id in doc_ids:
        process_one(doc_id)


if __name__ == "__main__":
    main()
