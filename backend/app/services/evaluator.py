from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from app.models.response_models import EvaluateScore

async def runRagas(question : str, answer : str, contexts: list[str]) -> EvaluateScore:
    try:
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [answer],
        })

        result = evaluate(dataset, metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ])

        row = result.to_pandas().iloc[0]

        return EvaluateScore(
            faithfulness=round(float(row.get("faithfulness", 0)), 4),
            answer_relevancy=round(float(row.get("answer_relevancy", 0)), 4),
            context_precision=round(float(row.get("context_precision", 0)), 4),
            context_recall=round(float(row.get("context_recall", 0)), 4),
        )
    except Exception:
        return EvaluateScore()