import json
from groq import Groq
from app.core.config import getSettings
from app.core.prompts import GROQ_JUDGE_PROMPT
from app.models.response_models import EvaluateScore

settings = getSettings()

def runGroqJudge(
    question : str,
    answer : str,
    contexts: list[str],
)-> tuple[EvaluateScore, str]:
        client = Groq(api_key=settings.GROQ_API_KEY)
        context_text = "\n\n---\n\n".join(contexts[:5])

        prompt = GROQ_JUDGE_PROMPT.format(
        question=question,
        context=context_text[:3000],
        answer=answer[:2000],
        )

        try:
             response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content" : prompt}],
                max_tokens=512,
                temperature=0.0

             )

             raw = response.choices[0].message.content.strip()
             if "``" in raw:
                raw = raw.split("``")[1].lstrip("json").strip()

             data = json.loads(raw)

             scores = EvaluateScore(
                       correctness=round(float(data.get("correctness", 0)), 4),
                       groundedness=round(float(data.get("groundedness", 0)), 4),
                       completeness=round(float(data.get("completeness", 0)), 4),
                       faithfulness=round(float(data.get("faithfulness", 0)), 4),
                       hallucination_risk=round(1.0 - float(data.get("faithfulness", 1)), 4),
             )
             return scores, data.get("reasoning", "")
        
        except Exception :
            return EvaluateScore(), "evaluatiion failed"