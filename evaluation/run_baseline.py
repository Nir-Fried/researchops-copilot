import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tracers.langchain import wait_for_all_tracers

from researchops_copilot.evaluation.metrics import (
    EvaluationCase,
    evaluate_case,
)
from researchops_copilot.models import (
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
)
from researchops_copilot.rag.chain import answer_question


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the dense-RAG baseline."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/dataset.json"),
        help="Path to the evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/results/dense-rag-baseline.json"
        ),
        help="Path where evaluation results will be written.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks retrieved for each question.",
    )

    return parser


def load_evaluation_cases(
    dataset_path: Path,
) -> list[EvaluationCase]:
    """Load and validate evaluation examples."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    raw_cases = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    return [
        EvaluationCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]


def safe_average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def create_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the individual evaluation results."""

    answerable_results = [
        result
        for result in results
        if result["expected_answerable"]
    ]

    return {
        "total_cases": len(results),
        "passed_cases": sum(
            result["passed"]
            for result in results
        ),
        "pass_rate": safe_average(
            [
                float(result["passed"])
                for result in results
            ]
        ),
        "answerability_accuracy": safe_average(
            [
                float(result["answerability_correct"])
                for result in results
            ]
        ),
        "retrieval_recall_at_k": safe_average(
            [
                float(result["retrieval_hit_at_k"])
                for result in answerable_results
            ]
        ),
        "citation_page_accuracy": safe_average(
            [
                float(result["citation_page_hit"])
                for result in answerable_results
            ]
        ),
        "average_term_coverage": safe_average(
            [
                result["term_coverage"]
                for result in answerable_results
            ]
        ),
    }


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    args = create_argument_parser().parse_args()

    if args.k <= 0:
        raise ValueError("--k must be greater than zero.")

    cases = load_evaluation_cases(args.dataset)
    results: list[dict[str, Any]] = []

    try:
        for index, case in enumerate(cases, start=1):
            print(
                f"\n[{index}/{len(cases)}] {case.id}"
            )
            print(f"Question: {case.question}")

            response = answer_question(
                question=case.question,
                k=args.k,
            )

            result = evaluate_case(
                case=case,
                response=response,
            )

            results.append(result)

            print(
                "Result:",
                "PASS" if result["passed"] else "FAIL",
            )
            print(
                "Answerability correct:",
                result["answerability_correct"],
            )

            if case.answerable:
                print(
                    f"Retrieval hit@{args.k}:",
                    result["retrieval_hit_at_k"],
                )
                print(
                    "Citation-page hit:",
                    result["citation_page_hit"],
                )
                print(
                    "Term coverage:",
                    format_percentage(
                        result["term_coverage"]
                    ),
                )

            print(f"Answer: {response.answer}")

    finally:
        wait_for_all_tracers()

    summary = create_summary(results)

    report = {
        "evaluation_name": "dense-rag-baseline",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": {
            "retriever": "chroma_dense_similarity",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chat_model": CHAT_MODEL_NAME,
            "k": args.k,
        },
        "summary": summary,
        "results": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    print("Evaluation summary")
    print(f"Cases: {summary['total_cases']}")
    print(
        "Passed:",
        f"{summary['passed_cases']}/"
        f"{summary['total_cases']}",
    )
    print(
        "Pass rate:",
        format_percentage(summary["pass_rate"]),
    )
    print(
        "Answerability accuracy:",
        format_percentage(
            summary["answerability_accuracy"]
        ),
    )
    print(
        f"Retrieval Recall@{args.k}:",
        format_percentage(
            summary["retrieval_recall_at_k"]
        ),
    )
    print(
        "Citation-page accuracy:",
        format_percentage(
            summary["citation_page_accuracy"]
        ),
    )
    print(
        "Average expected-term coverage:",
        format_percentage(
            summary["average_term_coverage"]
        ),
    )
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()