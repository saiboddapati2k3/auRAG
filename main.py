from fastapi import FastAPI, Header, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

from processor import load_and_index_document
from model import query_document

load_dotenv()

app = FastAPI()

# Setup header-based auth
api_key_header = APIKeyHeader(name="Authorization")

# Request schema
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_query(
    request: QueryRequest,
    authorization: str = Depends(api_key_header)
):
    # Optional auth check
    if not authorization.startswith("Bearer "):
        return {"status": "error", "message": "Invalid token format"}

    try:
        print("🚀 Loading document and indexing...")
        load_and_index_document(request.documents)

        print("💬 Answering questions...")
        results = []
        for q in request.questions:
            try:
                answer = query_document(q)
                results.append(answer)
            except Exception as qe:
                results.append(f"Error answering: {q} — {qe}")

        return {
            "status": "success",
            "answers": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Internal error: {str(e)}"
        }
