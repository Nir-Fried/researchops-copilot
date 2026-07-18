from researchops_copilot.agentic_rag.graph import (
    route_after_grading,
)


def test_route_generates_when_context_is_sufficient() -> None:
    state = {
        "sufficient_context": True,
        "relevant_documents": ["document"],
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_grading(state)
        == "generate_answer"
    )


def test_route_rewrites_when_context_is_insufficient() -> None:
    state = {
        "sufficient_context": False,
        "relevant_documents": [],
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_grading(state)
        == "rewrite_query"
    )


def test_route_abstains_when_retries_are_exhausted() -> None:
    state = {
        "sufficient_context": False,
        "relevant_documents": [],
        "retry_count": 1,
        "max_retries": 1,
    }

    assert (
        route_after_grading(state)
        == "abstain"
    )