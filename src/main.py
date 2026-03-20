# Sample main file

from fastapi import FastAPI
from .api import router

app = FastAPI(title="Q&A Dokumen Hukum")
app.include_router(router)
