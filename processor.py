import os
import requests
from dotenv import load_dotenv
import pinecone                                    # ← old SDK
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()

# ───────────────────────── Pinecone init ─────────────────────────
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV") or "us-east-1"
)
index_name = os.getenv("PINECONE_INDEX_NAME") or "aurag"

# create index if needed (1536 dims for OpenAI)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

embeddings = OpenAIEmbeddings()

# ───────────────────────── Helpers ─────────────────────────
def download_pdf(url: str, path: str = "temp_doc.pdf"):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

def load_and_index_document(doc_url: str):
    print("📥 Downloading...")
    download_pdf(doc_url)

    print("📄 Loading PDF...")
    loader = PyPDFLoader("temp_doc.pdf")
    docs = loader.load()

    print("🪓 Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks extracted from PDF.")

    print(f"✅ {len(chunks)} chunks")
    print("🔗 Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="default"
    )
    print("📦 Indexing complete.")
    return chunks
