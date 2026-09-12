"""
Tests for src.extraction.clause_extractor.

Fast tests: mock requests.post — verify prompt construction and JSON
parsing/validation logic without hitting a real Ollama instance.
Slow tests (@pytest.mark.slow): real Ollama call — verifies the actual
model follows the output-constraining instructions. Skips cleanly if
Ollama isn't running locally.
"""
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.extraction.clause_extractor import (
    extract_clauses,
    _parse_clauses,
    _call_ollama,
    _enforce_risk_rubric,
)


# ======== Fast Unit Tests — JSON Parsing / Validation ==================================

class TestParseClausesValidInput:
    def test_valid_json_parsed_correctly(self):
        raw = '{"clauses": [{"clause_num": "3.1", "text": "Banks shall notify RBI within 7 days.", "risk_level": "high", "reason": "hard deadline"}]}'
        result = _parse_clauses(raw)
        assert len(result) == 1
        assert result[0]["clause_num"] == "3.1"
        assert result[0]["risk_level"] == "high"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"clauses": [{"clause_num": "1", "text": "x", "risk_level": "low"}]}\n```'
        result = _parse_clauses(raw)
        assert len(result) == 1

    def test_risk_level_normalized_to_lowercase(self):
        raw = '{"clauses": [{"clause_num": "1", "text": "x", "risk_level": "HIGH"}]}'
        result = _parse_clauses(raw)
        assert result[0]["risk_level"] == "high"

    def test_missing_reason_defaults_empty_string(self):
        raw = '{"clauses": [{"clause_num": "1", "text": "x", "risk_level": "low"}]}'
        result = _parse_clauses(raw)
        assert result[0]["reason"] == ""


class TestParseClausesInvalidInput:
    def test_malformed_json_returns_empty_list(self):
        assert _parse_clauses("not valid json {{{") == []

    def test_invalid_risk_level_drops_that_clause_only(self):
        raw = (
            '{"clauses": ['
            '{"clause_num": "1", "text": "x", "risk_level": "critical"},'
            '{"clause_num": "2", "text": "y", "risk_level": "medium"}'
            "]}"
        )
        result = _parse_clauses(raw)
        assert len(result) == 2
        assert result[0]["risk_level"] == "invalid"
        assert result[1]["clause_num"] == "2"

    def test_missing_clause_num_assigns_placeholder(self):
        raw = '{"clauses": [{"text": "x", "risk_level": "low"}]}'
        result = _parse_clauses(raw)
        assert len(result) == 1
        assert result[0]["clause_num"] == "unnumbered_0"

    def test_empty_clauses_list_returns_empty(self):
        assert _parse_clauses('{"clauses": []}') == []

    def test_missing_clauses_key_returns_empty(self):
        assert _parse_clauses('{}') == []


# ======== Fast Unit Tests — Ollama Call Construction (mocked) ==================================

class TestCallOllamaMocked:
    @patch("src.extraction.clause_extractor.requests.post")
    def test_sends_correct_payload_shape(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "OK"}
        mock_post.return_value = mock_response

        result = _call_ollama("test prompt")

        assert result == "OK"
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["prompt"] == "test prompt"
        assert kwargs["json"]["stream"] is False
        assert kwargs["json"]["options"]["temperature"] == 0.0

    @patch("src.extraction.clause_extractor.requests.post")
    def test_raises_on_connection_failure(self, mock_post):
        """Ollama being unreachable must propagate, not be silently masked."""
        mock_post.side_effect = requests.ConnectionError("Ollama not running")
        with pytest.raises(requests.ConnectionError):
            _call_ollama("test prompt")


class TestExtractClausesEndToEndMocked:
    @patch("src.extraction.clause_extractor._call_ollama")
    def test_full_flow_with_mocked_llm_response(self, mock_call):
        mock_call.return_value = (
            '{"clauses": [{"clause_num": "1", "text": "Sample clause", '
            '"risk_level": "medium", "reason": "process obligation"}]}'
        )
        chapter_text = (
            "Chapter I - Sample\n"
            "This is a sample chapter used purely to exceed the 150 character "
            "minimum threshold so that extract_clauses() actually invokes the "
            "mocked Ollama call instead of short-circuiting on section length."
        )
        result = extract_clauses(chapter_text)
        assert len(result) == 1
        assert result[0]["risk_level"] == "medium"
        mock_call.assert_called_once()
        # Confirm the rubric was actually included in the prompt sent
        sent_prompt = mock_call.call_args[0][0]
        assert "high" in sent_prompt and "medium" in sent_prompt and "low" in sent_prompt


# ======== Slow Integration Test — Real Ollama ==================================

def _ollama_reachable() -> bool:
    try:
        from config.settings import settings
        requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


@pytest.mark.slow
class TestRealOllamaExtraction:
    def test_extracts_structured_clauses_from_sample_text(self):
        if not _ollama_reachable():
            pytest.skip("Ollama not reachable at configured OLLAMA_BASE_URL")

        sample_chapter = """
        Chapter II - Notice Requirements

        3.1 Banks shall notify the Reserve Bank within 7 days of any
        material change to credit card terms.

        3.2 "Cardholder" means any individual issued a credit card under
        these directions.
        """
        result = extract_clauses(sample_chapter)
        assert len(result) >= 1, "Expected at least one clause extracted"
        for clause in result:
            assert clause["risk_level"] in {"high", "medium", "low"}
            assert clause["clause_num"]
            assert clause["text"]

    def test_invalid_risk_level_drops_that_clause_only(self):
        raw = (
            '{"clauses": ['
            '{"clause_num": "1", "text": "x", "risk_level": "critical"},'
            '{"clause_num": "2", "text": "y", "risk_level": "medium"}'
            "]}"
        )
        result = _parse_clauses(raw)
        # No longer silently discarded — kept for audit, flagged via _invalid_risk.
        assert len(result) == 2
        assert result[0]["risk_level"] == "invalid"
        assert result[0]["_invalid_risk"] is True
        assert result[1]["clause_num"] == "2"
        assert "_invalid_risk" not in result[1]


class TestEnforceRiskRubricSkipsDroppedInvalid:
    def test_dropped_invalid_risk_not_overridden(self):
        clauses = [{
            "clause_num": "1", "text": "The bank shall comply.",
            "risk_level": "invalid", "reason": "", "status": "dropped_invalid_risk",
        }]
        result = _enforce_risk_rubric(clauses)
        assert result[0]["risk_level"] == "invalid"

class TestCheckOllamaReachable:
    @patch("src.extraction.clause_extractor.requests.get")
    def test_reachable_does_not_raise(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        from src.extraction.clause_extractor import check_ollama_reachable
        check_ollama_reachable()  # should not raise

    @patch("src.extraction.clause_extractor.requests.get")
    def test_unreachable_raises_runtime_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("refused")
        from src.extraction.clause_extractor import check_ollama_reachable
        import pytest
        with pytest.raises(RuntimeError, match="Ollama unreachable"):
            check_ollama_reachable()
