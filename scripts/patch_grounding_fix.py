"""
Hand-patches the 4 clauses confirmed to flip from dropped_ungrounded ->
included under the length-weighted grounding fix (see scripts/analyse_drops.py
output). Chosen over re-running reextract_sections.py to avoid introducing
LLM nondeterminism for a change already verified deterministically.

Run:
    python scripts/patch_grounding_fix.py
"""
import json
import os

INPUT_DIR = "data/extracted_clauses"

# (doc_id, clause_num, chapter_title) uniquely identifies each target clause.
PATCHES = [
    ("RBI_Credit_Debit_Card", "11.13",
     "Chapter II - Conduct of Credit Card Business :: 11.  Customer Acquisition:"),
    ("RBI_Digital_Payment_Security", "1",
     "Chapter VII - Repeal and Other Provisions :: A. Repeal and Saving"),
    ("RBI_Fraud_Risk_Management", "1",
     "Chapter X - Repeal and Other Provisions :: A. Repeal and Saving"),
    ("RBI_Managing_Risks", "25",
     "Chapter I - Preliminary :: B. Applicability and Scope"),
]


def patch_file(doc_id: str, clause_num: str, chapter_title: str) -> None:
    path = os.path.join(INPUT_DIR, f"{doc_id}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    matches = [
        c for c in data["clauses"]
        if c["clause_num"] == clause_num
        and c["chapter_title"] == chapter_title
        and c["status"] == "dropped_ungrounded"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{doc_id}: expected exactly 1 match for clause_num={clause_num!r} "
            f"chapter_title={chapter_title!r}, found {len(matches)} — aborting, no changes made."
        )

    matches[0]["status"] = "included"
    matches[0]["status_note"] = "flipped dropped_ungrounded->included via length-weighted grounding fix (analyse_drops.py verified)"

    def _count(status):
        return sum(1 for c in data["clauses"] if c["status"] == status)

    data["summary"]["included"] = _count("included")
    data["summary"]["dropped_ungrounded"] = _count("dropped_ungrounded")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{doc_id}: patched clause {clause_num!r} -> included. "
          f"New summary: included={data['summary']['included']}, "
          f"dropped_ungrounded={data['summary']['dropped_ungrounded']}")


def main():
    for doc_id, clause_num, chapter_title in PATCHES:
        patch_file(doc_id, clause_num, chapter_title)


if __name__ == "__main__":
    main()
