from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from app.models.request_models import IngestReq
from app.models.response_models import IngestResponse
from app.services.ingest_service import ingestRepo
from app.api.deps import getChunkStore

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestReq):
    try : 
        result = ingestRepo(request.repoUrl, request.branch)

        chunkstore = getChunkStore()
        chunkstore[result["namespace"]] = result["chunks"]

        return IngestResponse(
            success = True,
            repoUrl = request.repoUrl,
            chunksIndexed = result["chunksIndexed"],
            filesProcessed = result["filesProcessed"],
            message = f"indexed{result["filesProcessed"]} files, {result['chunksIndexed']} chunks"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))