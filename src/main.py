# Sample main file

from fastapi import FastAPI
from .api import router

app = FastAPI(title="Document Q&A with AI")
app.include_router(router)
