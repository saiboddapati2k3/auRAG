import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Pinecone init (new method)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
env_region = os.getenv("PINECONE_ENV")  # e.g. aped-4627-b74a

# Create the index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # depends on your embedding model (e.g., OpenAI)
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # use your actual region (from Pinecone dashboard)
        )
    )

embeddings = OpenAIEmbeddings()

def load_and_index_document(doc_url):
    # Download the document
    r = requests.get(doc_url)
    with open("temp_doc.pdf", "wb") as f:
        f.write(r.content)

    loader = PyPDFLoader("temp_doc.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create vectorstore and store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="default",  # optional, can separate by file/query
    )

    return chunks
