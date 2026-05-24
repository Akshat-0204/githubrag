from pydantic import BaseModel
from typing import Optional 

class IngestResponse(BaseModel):
    success : bool
    repoUrl : str
    chunksIndexed : int
    filesProcessed : int
    message : str

class SourceChunk(BaseModel):
    filePath : str
    content : str
    language : str
    score : float 

class EvaluateScore(BaseModel):
    faithfulness: Optional[float] = None
    answerRelevancy: Optional[float] = None
    contextPrecision: Optional[float] = None
    contextRecall: Optional[float] = None
    correctness: Optional[float] = None
    groundedness: Optional[float] = None
    completeness: Optional[float] = None
    hallucinationRisk: Optional[float] = None

class EvaluateResponse(BaseModel):
    question: str
    scores : EvaluateScore
    reasoning : Optional[str] = None
    passed : bool