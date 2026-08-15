"""
Neo4j connection client — thin wrapper over the official neo4j driver.

Design:
- One Neo4jClient instance per application lifetime, wrapping ONE driver
  (expensive to create, safe to reuse — driver).
- Sessions are opened per call, short-lived, closed immediately after
- All writes/reads go through managed transactions (execute_write /
  execute_read) rather than session.run() directly — automatic retry
  on transient errors (network blips, leader re-election on clustered
  deployments).
- All queries MUST use parameterized Cypher — never string-format
  values into a query (same discipline as parameterized SQL).
- IMPORTANT: functions passed to execute_read/execute_write must contain
  ONLY Cypher calls — no side effects outside the transaction (API calls,
  file writes, etc.), because a transient-error retry re-runs the whole
  function from scratch.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from neo4j import GraphDatabase, Driver, ManagedTransaction

from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Wraps a single Neo4j Driver instance for the application's lifetime.

    Usage:
        client = Neo4jClient()
        client.verify_connectivity()
        result = client.execute_read(my_read_fn, param=value)
        client.close()

    Or as a context manager:
        with Neo4jClient() as client:
            client.execute_write(my_write_fn, param=value)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        settings.validate()  # fails fast if NEO4J_URI/PASSWORD missing from .env

        self._uri = uri or settings.NEO4J_URI
        self._username = username or settings.NEO4J_USERNAME
        self._password = password or settings.NEO4J_PASSWORD

        self._driver: Driver = GraphDatabase.driver(
            self._uri, auth=(self._username, self._password)
        )
        logger.info("Neo4j driver created for %s", self._uri)

    def verify_connectivity(self) -> None:
        """
        Confirms the driver can actually reach and authenticate against
        the Neo4j instance. Call once at startup — fails fast and loud
        instead of letting the first real query surprise you with a
        connection error deep inside a pipeline run.
        """
        self._driver.verify_connectivity()
        logger.info("Neo4j connectivity verified")

    def execute_read(
        self, work: Callable[[ManagedTransaction], Any], **params: Any
    ) -> Any:
        """
        Runs `work` inside a managed READ transaction. Neo4j Aura may
        route reads to a replica; execute_read (vs execute_write) lets
        the driver make that routing decision correctly.
        """
        with self._driver.session() as session:
            return session.execute_read(work, **params)

    def execute_write(
        self, work: Callable[[ManagedTransaction], Any], **params: Any
    ) -> Any:
        """
        Runs `work` inside a managed WRITE transaction. A retried write
        must be safe to re-run — prefer MERGE over blind CREATE for
        anything that shouldn't duplicate on retry.
        """
        with self._driver.session() as session:
            return session.execute_write(work, **params)

    def close(self) -> None:
        """Closes the driver's connection pool. Call once at app shutdown."""
        self._driver.close()
        logger.info("Neo4j driver closed")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
