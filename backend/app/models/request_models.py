from pydantic import BaseModel
from typing import Optional 

class IngestReq(BaseModel):
    repoUrl : str
    branch : str = "main"

class ChatReq(BaseModel):
    question : str
    repoUrl : str

class EvaluateReq(BaseModel):
    question : str
    answer : str
    contexts : list[str]