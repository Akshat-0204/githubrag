from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.request_models import ChatReq
from app.services.rag_pipeline import runRagStream
from app.api.deps import getChunkStore

router = APIRouter()

@router.post("/chat")
async def chat(request : ChatReq):
    namespace = request.repo_url.rstrip("/").replace("https://github.com/", "").replace("/", "_")
    chunkstore = getChunkStore()

    if namespace not in chunkstore:
        raise HTTPException(status_code=400, detail="Repo not ingested uet")

    return StreamingResponse(
        runRagStream(
            question=request.question,
            namespace=namespace,
            chunks=chunkstore[namespace]
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},

    )