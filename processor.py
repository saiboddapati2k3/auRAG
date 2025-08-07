import os
import requests
import hashlib
from typing import List, Set
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

class OptimizedDocumentProcessor:
    def __init__(self):
        self.pc = Pinecone()
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "aurag")
        self.pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self._ensure_index_exists()
        
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=PodSpec(environment=self.pinecone_env)
            )
            
    def _download_pdf(self, url: str, path: str = "temp_doc.pdf") -> str:
        """Download PDF with error handling"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path
        except Exception as e:
            raise Exception(f"Failed to download PDF: {str(e)}")
    
    def _smart_chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Advanced chunking with overlapping windows and semantic boundaries"""
        # Primary splitter - larger chunks with good overlap
        primary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True
        )
        
        # Secondary splitter for very long texts
        secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n"],
            keep_separator=True
        )
        
        all_chunks = []
        
        for doc in docs:
            # Use different strategies based on document length
            if len(doc.page_content) > 5000:
                # For long documents, use hierarchical chunking
                large_chunks = secondary_splitter.split_documents([doc])
                for large_chunk in large_chunks:
                    small_chunks = primary_splitter.split_documents([large_chunk])
                    all_chunks.extend(small_chunks)
            else:
                # For shorter documents, use primary splitter
                chunks = primary_splitter.split_documents([doc])
                all_chunks.extend(chunks)
        
        return self._deduplicate_chunks(all_chunks)
    
    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove duplicate chunks based on content similarity"""
        seen_hashes: Set[str] = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create hash of normalized content
            normalized_content = " ".join(chunk.page_content.split()).lower()
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def load_and_index_document(self, doc_url: str) -> List[Document]:
        """Optimized document loading and indexing"""
        pdf_path = self._download_pdf(doc_url)
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            if not docs:
                raise ValueError("No content extracted from PDF")
            
            chunks = self._smart_chunk_documents(docs)
            
            if not chunks:
                raise ValueError("No text chunks created from document")
            
            # Batch upload with error handling
            try:
                PineconeVectorStore.from_documents(
                    chunks,
                    embedding=self.embeddings,
                    index_name=self.index_name,
                    namespace="default",
                    batch_size=16  # Smaller batches for stability
                )
                return chunks
            except Exception as e:
                raise
                
        finally:
            # Cleanup
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
    def delete_all_from_index(self):
        """Clear the index"""
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True, namespace="default")

# Global instance
processor = OptimizedDocumentProcessor()
load_and_index_document = processor.load_and_index_document
delete_all_from_index = processor.delete_all_from_index