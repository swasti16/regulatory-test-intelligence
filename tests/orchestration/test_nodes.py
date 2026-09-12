from src.orchestration.nodes import validate_node


def _base_clause(**overrides):
    clause = {
        "clause_num": "1",
        "text": "Banks shall notify RBI within 7 days of any material change.",
        "risk_level": "high",
        "chapter_title": "Chapter I - Preliminary",
        "page_start": 1,
        "status": "included",
        "truncated": False,
    }
    clause.update(overrides)
    return clause


def _base_state(clauses):
    return {
        "doc_id": "TEST_DOC",
        "all_clauses": clauses,
        "validation_issues": [],
    }


class TestValidateNodeTruncatedWarning:
    def test_truncated_clause_produces_warn(self):
        state = _base_state([_base_clause(truncated=True)])
        result = validate_node(state)
        issues = result["validation_issues"]

        warns = [i for i in issues if "truncated" in i.lower()]
        fails = [i for i in issues if i.startswith("FAIL:")]

        assert len(warns) == 1
        assert "clause_num=1" in warns[0]
        assert fails == []

    def test_non_truncated_clause_no_warning(self):
        state = _base_state([_base_clause(truncated=False)])
        result = validate_node(state)
        assert not any("truncated" in i.lower() for i in result["validation_issues"])
