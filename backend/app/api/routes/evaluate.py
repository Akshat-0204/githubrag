from fastapi import APIRouter, HTTPException

from app.models.request_models import EvaluateReq
from app.models.response_models import EvaluateResponse, EvaluateScore
from app.services.evaluator import runRagas
from app.services.groq_judge import runGroqJudge

router = APIRouter()

@router.post("/evaluate", response_model= EvaluateResponse)
async def evaluate(request: EvaluateReq):
    try:
        ragasScore = await runRagas(request.question, request.answer, request.contexts)
        groqScore, reasoning = runGroqJudge(request.question,request.answer, request.contexts )

        merged = EvaluateScore(
            faithfulness=ragasScore.faithfulness or groqScore.faithfulness,
            answer_relevancy=ragasScore.answer_relevancy,
            context_precision=ragasScore.context_precision,
            context_recall=ragasScore.context_recall,
            correctness=groqScore.correctness,
            groundedness=groqScore.groundedness,
            completeness=groqScore.completeness,
            hallucination_risk=groqScore.hallucination_risk,
        )

        key = [s for s in [merged.faithfulness, merged.correctness, merged.groundedness] if s ]
        passed = (sum(key) / len(key)) >= 0.6 if key else False

        return EvaluateResponse(
            question=request.question,
            scores=merged,
            reasoning=reasoning,
            passed=passed,

        )

    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))