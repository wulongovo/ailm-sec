"""Unit tests for AILLM-SEC API."""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    from core.embeddings import EmbeddingService
    from core.rag import KnowledgeBase
    from core.llm import LLMEngine
    from core.ingestion import DocumentIngestor
    assert True


def test_embedding_service_init():
    """Test embedding service initialization."""
    from core.embeddings import EmbeddingService
    svc = EmbeddingService(provider="ollama", model="bge-m3")
    assert svc.provider == "ollama"
    assert svc.model == "bge-m3"


def test_embedding_dummy():
    """Test dummy embedding when no provider available."""
    from core.embeddings import EmbeddingService
    svc = EmbeddingService(provider="none")
    result = svc.embed_query("test")
    assert len(result) == 384


def test_ingestion_format_cve():
    """Test CVE formatting."""
    from core.ingestion import DocumentIngestor
    ingestor = DocumentIngestor(None)
    item = {
        "id": "CVE-2021-44228",
        "description": "Log4j JNDI injection",
        "severity": "critical",
    }
    text = ingestor._format_cve(item)
    assert "CVE-2021-44228" in text
    assert "Log4j" in text


@pytest.fixture
def mock_knowledge_base():
    """Mock knowledge base for testing."""
    from unittest.mock import MagicMock
    kb = MagicMock()
    kb.get_stats.return_value = {"status": "ready", "documents": 100}
    kb.query.return_value = [{"content": "test doc", "metadata": {"source": "test"}}]
    return kb


def test_llm_engine_init():
    """Test LLM engine initialization."""
    from core.llm import LLMEngine
    engine = LLMEngine()
    assert engine.model is not None


def test_api_health():
    """Test health endpoint."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    except ImportError:
        pytest.skip("TestClient not available")
