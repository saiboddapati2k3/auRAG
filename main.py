from fastapi import FastAPI, Header, Depends
from fastapi.security.api_key import APIKeyHeader
from processor import load_and_index_document
from model import query_document
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

api_key_header = APIKeyHeader(name="Authorization")

@app.post("/hackrx/run")
async def run_query(
    request: QueryRequest,
    authorization: str = Depends(api_key_header)
):
    if not authorization.startswith("Bearer "):
        return {"status": "error", "message": "Invalid token format"}

    # Step 1: Load and embed document
    try:
        doc_chunks = load_and_index_document(request.documents)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # Step 2: Process each question
    results = []
    for q in request.questions:
        answer = query_document(q)
        results.append(answer)

    return {
        "status": "success",
        "answers": results
    }
