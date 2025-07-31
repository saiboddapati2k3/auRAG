import os
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from Utils.doc_loader import load_pdf_from_url
from Utils.rag_pipeline import get_qa_chain
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Security check
REQUIRED_TOKEN = "Bearer YOUR_EXPECTED_API_KEY"  # Replace with your expected test token

class QueryPayload(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def process_hackrx(request: Request, payload: QueryPayload, authorization: str = Header(None)):
    if authorization != REQUIRED_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Bearer Token")

    try:
        document_text = load_pdf_from_url(payload.documents)
        qa_chain = get_qa_chain(document_text)

        results = []
        for q in payload.questions:
            answer = qa_chain.run(q)
            results.append(answer)

        return {"answers": results}

    except Exception as e:
        return {"answers": [], "error": str(e)}
