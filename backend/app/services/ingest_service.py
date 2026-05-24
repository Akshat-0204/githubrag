import os 
import shutil 
import tempfile 
from pathlib import Path

import git
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec 

from app.core.config import getSettings

settings = getSettings()

SUPPORTED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
    ".java", ".cpp", ".c", ".cs", ".rb", ".md", ".yaml",
    ".yml", ".json", ".toml", ".sh", ".sql", ".graphql",
}

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "coverage", "target", "vendor",
}

LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS,
    ".ts": Language.JS, ".tsx": Language.JS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
}

EXT_TO_LANG = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".cs": "csharp", ".rb": "ruby", ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".sh": "bash", ".sql": "sql", ".graphql": "graphql",
}

def ensurePineConeIndex():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    exisiting = [i.name for i in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME not in exisiting:
        pc.create_index(
            name = settings.PINECONE_INDEX_NAME,
            dimension=1536,
            metric = "cosine",
            spec=ServerlessSpec(cloud="aws" , region="us-east-1"),
        )
    return pc

def cloneRepo(repoUrl: str, branch : str) -> str:
    temp_dir = tempfile.mkdtemp()
    try:
        git.Repo.clone_from(repoUrl, temp_dir, branch=branch, depth=1)
    except git.exc.GitCommandError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir = tempfile.mkdtemp()
        git.Repo.clone_from(repoUrl, temp_dir, depth=1)
    return temp_dir

def loadDocuments(repoDir : str, repoUrl : str) -> list[Document]:
    docs = []
    root = Path(repoDir)

    for filePath in root.rglob("*"):
        if not filePath.is_file():
            continue

        relative = filePath.relative_to(root)

        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
    
        if filePath.suffix not in SUPPORTED_EXTENSIONS:
            continue

        if filePath.stat().st_size > 200_000:
            continue

        try : 
            content = filePath.read_text(encoding="utf-8", errors="ignore").strip()
            if not content : 
                continue

            docs.append(Document(
                page_content=content,
                metadata = {
                    "filePath" : str(relative),
                    "language" : EXT_TO_LANG.get(filePath.suffix, "text"),
                    "repoUrl" : repoUrl 
                }
            ))

        except Exception :
            continue
    return docs


#Document chunking 

def chunkDocs(docs: list[Document]) -> list[Document]:
    allChunks = []

    for doc in docs:
        ext = "." + doc.metadata["filePath"].split(".")[-1]
        lang = LANGUAGE_MAP.get(ext)

        try :
            if lang:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                )
            else : 
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                )
            chunks = splitter.split_documents([doc])
            allChunks.extend(chunks)
        except Exception:
            continue
    
    return allChunks

def ingestRepo(repoUrl: str, branch : str = "main") -> dict :
    pc = ensurePineConeIndex()
    namespace = repoUrl.rstrip("/").replace("https://github.com/","").replace("/", "_")

    #if old data , clear old data
    try:
        pc.Index(settings.PINECONE_INDEX_NAME).delete(delete_all=True , namespace=namespace)
    except Exception:
        pass

    #Clone
    temp_dir = cloneRepo(repoUrl, branch)

    try:
        #load the doc
        docs = loadDocuments(temp_dir, repoUrl)
        if not docs:
            raise ValueError("No supported files found in repo")

        #chunkig 
        chunks = chunkDocs(docs)

        #Embed and store embeds in pinecone
        embeddings = OpenAIEmbeddings(
            model = settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key= settings.OPENAI_API_KEY,
        )

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name = settings.PINECONE_INDEX_NAME,
            namespace = namespace
        )

        return {
            "namespace" : namespace,
            "filesProcessed" : len(docs),
            "chunksIndexed" : len(chunks),
            "chunks" : chunks
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)