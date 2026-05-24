from langchain_core.documents import Document

chunkStore: dict[str, list[Document]] = {}

def getChunkStore() -> dict[str, list[Document]]:
    return chunkStore


