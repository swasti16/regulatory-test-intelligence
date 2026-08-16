"""
Tests for src.graph.neo4j_client.Neo4jClient.

All tests mock neo4j.GraphDatabase.driver — no real Neo4j Aura connection
needed. Fast, deterministic, runs in CI without credentials.
"""
from unittest.mock import MagicMock, patch
import pytest

from src.graph.neo4j_client import Neo4jClient


# ======== Fixtures ==================================

@pytest.fixture(autouse=True)
def mock_settings():
    """Patches config.settings.settings so no real .env is required.
    autouse=True: every test in this module constructs a Neo4jClient(),
    which always calls settings.validate() — no test should ever run
    against real settings.
    """
    with patch("src.graph.neo4j_client.settings") as mock_s:
        mock_s.NEO4J_URI = "neo4j+s://fake-instance.databases.neo4j.io"
        mock_s.NEO4J_USERNAME = "neo4j"
        mock_s.NEO4J_PASSWORD = "fake_password"
        mock_s.validate.return_value = None
        yield mock_s


@pytest.fixture
def mock_driver():
    """A MagicMock standing in for the real neo4j Driver instance."""
    driver = MagicMock()
    mock_session = MagicMock()
    mock_tx = MagicMock()

    mock_session.execute_read.side_effect = lambda work, **params: work(mock_tx, **params)
    mock_session.execute_write.side_effect = lambda work, **params: work(mock_tx, **params)

    driver.session.return_value.__enter__.return_value = mock_session
    driver.session.return_value.__exit__.return_value = False
    return driver


@pytest.fixture(autouse=True)
def mock_graph_database(mock_driver):
    """Patches neo4j.GraphDatabase.driver(...) to return our fake driver.
    autouse=True: every test constructs a Neo4jClient(), which always
    calls GraphDatabase.driver() — no test should ever attempt a real
    network connection to Aura.
    """
    with patch("src.graph.neo4j_client.GraphDatabase") as mock_gd:
        mock_gd.driver.return_value = mock_driver
        yield mock_gd

# ======== Driver Creation ==================================

class TestClientInitialization:
    def test_creates_driver_with_correct_uri_and_auth(self, mock_graph_database):
        Neo4jClient()
        mock_graph_database.driver.assert_called_once_with(
            "neo4j+s://fake-instance.databases.neo4j.io",
            auth=("neo4j", "fake_password"),
        )

    def test_validates_settings_before_connecting(self, mock_settings):
        Neo4jClient()
        mock_settings.validate.assert_called_once()

    def test_explicit_args_override_settings(self, mock_graph_database):
        Neo4jClient(uri="neo4j+s://other.databases.neo4j.io",
                   username="other_user", password="other_pass")
        mock_graph_database.driver.assert_called_once_with(
            "neo4j+s://other.databases.neo4j.io",
            auth=("other_user", "other_pass"),
        )


# ======== Connectivity ==================================

class TestVerifyConnectivity:
    def test_calls_driver_verify_connectivity(self, mock_driver):
        client = Neo4jClient()
        client.verify_connectivity()
        mock_driver.verify_connectivity.assert_called_once()

    def test_propagates_connection_failure(self, mock_driver):
        """
        Connectivity failures (bad credentials, unreachable Aura instance)
        must surface to the caller, not be silently swallowed — fail
        fast and loud at startup, per the docstring's design intent.
        """
        mock_driver.verify_connectivity.side_effect = ConnectionError(
            "Unable to reach Neo4j"
        )
        client = Neo4jClient()
        with pytest.raises(ConnectionError):
            client.verify_connectivity()


# ======== Read / Write Execution ==================================

class TestExecuteRead:
    def test_opens_session_and_calls_execute_read(self, mock_driver):
        client = Neo4jClient()
        fake_work_fn = MagicMock(return_value="read_result")

        result = client.execute_read(fake_work_fn, doc_id="155MD")

        mock_driver.session.assert_called_once()
        session = mock_driver.session.return_value.__enter__.return_value
        session.execute_read.assert_called_once_with(fake_work_fn, doc_id="155MD")
        assert result == "read_result"


class TestExecuteWrite:
    def test_opens_session_and_calls_execute_write(self, mock_driver):
        client = Neo4jClient()
        fake_work_fn = MagicMock(return_value="write_result")

        result = client.execute_write(fake_work_fn, doc_id="155MD", title="Test Reg")

        mock_driver.session.assert_called_once()
        session = mock_driver.session.return_value.__enter__.return_value
        session.execute_write.assert_called_once_with(
            fake_work_fn, doc_id="155MD", title="Test Reg"
        )
        assert result == "write_result"

    def test_new_session_opened_per_call(self, mock_driver):
        """
        Sessions are short-lived and opened per call, never reused
        across multiple execute_write/execute_read invocations — this
        locks in that design so a future refactor doesn't accidentally
        start holding sessions open long-term.
        """
        client = Neo4jClient()
        fn = MagicMock()

        client.execute_write(fn, a=1)
        client.execute_write(fn, a=2)

        assert mock_driver.session.call_count == 2


# ======== Lifecycle — close() and context manager ==================================

class TestClientLifecycle:
    def test_close_closes_driver(self, mock_driver):
        client = Neo4jClient()
        client.close()
        mock_driver.close.assert_called_once()

    def test_context_manager_closes_driver_on_exit(self, mock_driver):
        with Neo4jClient() as client:
            assert isinstance(client, Neo4jClient)
        mock_driver.close.assert_called_once()

    def test_context_manager_closes_driver_even_on_exception(self, mock_driver):
        """
        __exit__ must close the driver even if an exception was raised
        inside the `with` block — same guarantee as a Playwright
        BrowserContext closing on test failure, not just on success.
        """
        with pytest.raises(ValueError):
            with Neo4jClient() as _:
                raise ValueError("simulated failure inside the block")
        mock_driver.close.assert_called_once()
