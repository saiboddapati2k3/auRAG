import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
region = os.getenv("PINECONE_ENV") or "us-east-1"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=region
        )
    )

embeddings = OpenAIEmbeddings()


def download_pdf(doc_url: str, save_path: str = "temp_doc.pdf"):
    """Downloads PDF from a URL and saves it locally."""
    try:
        response = requests.get(doc_url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download PDF: {e}")

    with open(save_path, "wb") as f:
        f.write(response.content)


def load_and_index_document(doc_url: str):
    """Loads, splits, embeds, and uploads a PDF to Pinecone vector store."""
    # Step 1: Download PDF
    print("📥 Downloading document...")
    download_pdf(doc_url)

    # Step 2: Load PDF
    print("📄 Loading PDF...")
    try:
        loader = PyPDFLoader("temp_doc.pdf")
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"PDF load error: {e}")

    # Step 3: Split into chunks
    print("🪓 Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No chunks were generated from the document.")

    print(f"✅ Total chunks created: {len(chunks)}")

    # Step 4: Embed & store in Pinecone
    print("🔗 Uploading chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="default"
    )

    print("📦 Document successfully indexed.")
    return chunks
