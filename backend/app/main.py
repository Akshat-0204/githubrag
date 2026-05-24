from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import ingest, chat, evaluate

app = FastAPI(title="githubrag")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],

)

app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(evaluate.router, prefix="/api", tags=["Evaluate"])


