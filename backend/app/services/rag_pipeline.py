import json
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from app.core.config import getSettings
from app.core.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from app.services.retriever import buildRetriever , rerank

settings = getSettings()

def formatContext(docs : list[Document]) -> str:
    parts = []
    for i , doc in enumerate(docs, 1):
        fp = doc.metadata.get("filePath" , "unknown")
        lang = doc.metadata.get("language", "text")
        parts.append(f"[{i}] File: {fp}\n```{lang}\n{doc.page_content}\n```")
    return"\n\n---\n\n".join(parts)

def serverSentEvent(eventType: str, data : dict) -> str:
    return f"data:{json.dumps({'type': eventType, **data})}\n\n"

async def runRagStream(
    question: str,
    namespace: str,
    chunks: list[Document],
) -> AsyncGenerator[str, None]:
    llm = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
        temperature=0.1,
    )

    #retrieve
    yield serverSentEvent("status" , {"message" : "Retrieving code.."})
    retriever = buildRetriever(chunks, namespace)
    retrieved = retriever.invoke(question)

    #rerank
    yield serverSentEvent("status", {"message": "Reranking results..."})
    reranked = rerank(question, retrieved)

    #send source
    sources = [
        {
            "filePath": doc.metadata.get("filePath", ""),
            "content": doc.page_content[:400],
            "language": doc.metadata.get("language", "text"),
            "score": round(1.0 - (i * 0.05), 2),
        }
        for i, doc in enumerate(reranked)
    ]
    yield serverSentEvent("sources", {"sources": sources})

    #generate 
    yield serverSentEvent("status", {"message": "Generating answer..."})
    context = formatContext(reranked)
    message = [
        SystemMessage(content= RAG_SYSTEM_PROMPT),
        HumanMessage(content=RAG_USER_PROMPT.format(
                context = context,
                question=question,
        ))
    ]

    async for chunk in llm.astream(message):
        if chunk.content:
            yield serverSentEvent("token", {"content" : chunk.content})

    yield f"data: {json.dumps({"type" : 'done'})} \n\n"