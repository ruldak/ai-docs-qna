# Sample main file

from fastapi import FastAPI
from .api import router

app = FastAPI(title="Q&A Document with AI")
app.include_router(router)
