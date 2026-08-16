"""
Central configuration for Regulatory Test Intelligence.
All settings loaded from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    # ======== Neo4j Aura ==================================
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # ======== Ollama (teammate-hosted) ==================================
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # ======== Rule Thresholds ==================================
    LOW_COVERAGE_THRESHOLD: int = int(os.getenv("LOW_COVERAGE_THRESHOLD", "80"))

    # ======== Logging ==================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    SAMPLE_PDF_DIR = "data/sample_regulations"
    RBI_PDF_DIR: str = "data/RBI_regulations"

    def validate(self) -> None:
        """Validate required settings are present before pipeline runs."""
        missing = []
        if not self.NEO4J_URI:
            missing.append("NEO4J_URI")
        if not self.NEO4J_PASSWORD:
            missing.append("NEO4J_PASSWORD")
        if missing:
            raise RuntimeError(f"Missing required settings: {', '.join(missing)}")


settings = Settings()
