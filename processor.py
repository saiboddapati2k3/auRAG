import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec  # ← Updated import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore # ← Updated import

load_dotenv()

# ───────────────────────── Pinecone init (Updated) ─────────────────────────
# The new client automatically uses PINECONE_API_KEY from .env
pc = Pinecone()

# Get index name and environment from .env, with fallbacks
index_name = os.getenv("PINECONE_INDEX_NAME") or "aurag"
pinecone_env = os.getenv("PINECONE_ENV") or "us-east-1"

# Create index if it doesn't exist (1536 dims for OpenAI)
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimension for OpenAI's text-embedding-ada-002
        metric="cosine",
        spec=PodSpec(environment=pinecone_env)
    )

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
    # This LangChain function works with the new SDK via the langchain-pinecone package
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="default"
    )
    print("📦 Indexing complete.")
    return chunks