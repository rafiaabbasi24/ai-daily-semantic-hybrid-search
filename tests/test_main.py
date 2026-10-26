import json
from typing import List, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import the FastAPI app. Adjust the import path according to the actual project layout.
# It is assumed that the main FastAPI instance is exported as `app` from `app.main`.
from app.main import app  # type: ignore

# Import the function that performs the hybrid search. The concrete location may differ;
# update the import if necessary.
from app.search import hybrid_search  # type: ignore


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """
    Provide a synchronous TestClient for simple endpoint testing.
    """
    return TestClient(app)


@pytest.fixture(scope="module")
async def async_client() -> AsyncClient:
    """
    Provide an asynchronous HTTP client for testing async endpoints.
    """
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture(autouse=True)
def mock_hybrid_search(monkeypatch):
    """
    Patch `hybrid_search` with a deterministic implementation to avoid loading
    heavy models during testing.

    The mock returns a fixed list of results sorted by a decreasing score.
    """
    def _mock_hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
        # Simple deterministic data based on the query length.
        base_score = len(query) / 100.0
        results = [
            {
                "id": f"doc_{i}",
                "text": f"Result {i} for '{query}'",
                "score": round(base_score - i * 0.01, 4),
            }
            for i in range(top_k)
        ]
        return results

    monkeypatch.setattr("app.search.hybrid_search", _mock_hybrid_search)


def test_health_endpoint(test_client: TestClient) -> None:
    """
    Verify that the health check endpoint returns a 200 status and the expected payload.
    """
    response = test_client.get("/health")
    assert response.status_code == 200, "Health endpoint should return HTTP 200"
    data = response.json()
    assert isinstance(data, dict), "Health response must be a JSON object"
    assert data.get("status") == "ok", "Health response should contain status='ok'"


def test_search_endpoint_sync(test_client: TestClient) -> None:
    """
    Test the synchronous `/search` endpoint using a POST request.
    The endpoint is expected to accept a JSON body with a `query` field and return
    a JSON object containing a `results` list.
    """
    query = "test query"
    response = test_client.post(
        "/search",
        json={"query": query, "top_k": 3},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200, "Search endpoint should return HTTP 200"

    payload = response.json()
    assert isinstance(payload, dict), "Response must be a JSON object"
    assert "results" in payload, "Response JSON must contain a 'results' key"

    results = payload["results"]
    assert isinstance(results, list), "'results' must be a list"
    assert len(results) == 3, "Number of returned results should match `top_k`"

    # Verify the structure of each result.
    for i, result in enumerate(results):
        assert isinstance(result, dict), "Each result must be a dict"
        assert result["id"] == f"doc_{i}", "Result IDs should follow the mock pattern"
        expected_text = f"Result {i} for '{query}'"
        assert result["text"] == expected_text, "Result text does not match expected format"
        # Score should be decreasing.
        if i > 0:
            assert result["score"] < results[i - 1]["score"], "Scores should be sorted descending"


@pytest.mark.asyncio
async def test_search_endpoint_async(async_client: AsyncClient) -> None:
    """
    Test the asynchronous `/search` endpoint using an async client.
    This mirrors the synchronous test but validates async handling.
    """
    query = "async test"
    response = await async_client.post(
        "/search",
        json={"query": query, "top_k": 2},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200, "Async search endpoint should return HTTP 200"

    payload = response.json()
    assert isinstance(payload, dict)
    assert "results" in payload

    results = payload["results"]
    assert isinstance(results, list)
    assert len(results) == 2

    # Validate deterministic mock output.
    for i, result in enumerate(results):
        assert result["id"] == f"doc_{i}"
        assert result["text"] == f"Result {i} for '{query}'"


def test_hybrid_search_function_direct() -> None:
    """
    Directly test the `hybrid_search` function (which has been monkey‑patched
    by the fixture) to ensure it returns data in the expected format.
    """
    query = "direct function call"
    top_k = 4
    results = hybrid_search(query, top_k=top_k)

    assert isinstance(results, list), "hybrid_search should return a list"
    assert len(results) == top_k, "Number of results should match `top_k`"

    previous_score = None
    for i, item in enumerate(results):
        assert isinstance(item, dict), "Each result should be a dict"
        assert "id" in item and "text" in item and "score" in item, "Missing required keys"
        assert item["id"] == f"doc_{i}"
        assert item["text"] == f"Result {i} for '{query}'"
        assert isinstance(item["score"], float), "Score must be a float"
        if previous_score is not None:
            assert item["score"] < previous_score, "Scores must be in descending order"
        previous_score = item["score"]