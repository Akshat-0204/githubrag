from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
import cohere

from app.core.config import getSettings

settings = getSettings()

def buildRetriever(chunks: list[Document], namespace : str) -> EnsembleRetriever:
    #embeddings 
    embeddings = OpenAIEmbeddings(
        model = settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key = settings.OPENAI_API_KEY
    )

    #hybrid search karna hai 
    #dense search- pinecone
    vectorstore = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding= embeddings ,
        namespace= namespace,

    )

    dense = vectorstore.as_retriever(search_kwargs={"k" : settings.DENSE_TOP_K})

    #sparse 
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = settings.SPARSE_TOP_K

    #merging
    return EnsembleRetriever(
        retrievers = [dense, sparse],
        weights = [0.6 , 0.4]
    )

def rerank(query : str, docs : list[Document]) -> list[Document]:
    if not docs :
        return []
    
    co = cohere.Client(
        api_key=settings.COHERE_API_KEY
    )

    texts = [doc.page_content for doc in docs]

    response = co.rerank(
        model = "rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=settings.RERANK_TOP_N
    )

    return [docs[r.index] for r in response.results]

    