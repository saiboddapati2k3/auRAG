import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel

class Queries(BaseModel):
    """A list of search queries."""
    queries: List[str] = Field(description="A list of search queries.")

# Enhanced query generation prompt
QUERY_GENERATION_PROMPT = """
# ROLE
You are an expert search query strategist, specializing in formulating queries for dense, unstructured legal and corporate documents like policy wordings, contracts, and internal emails.

# DOCUMENT CONTEXT
The search will be performed on a corpus of formal documents that contain:
- Legal and technical jargon (e.g., "indemnification," "force majeure," "subrogation").
- Specific clause or section numbering (e.g., "Clause 4.1.b," "Section III, Part A").
- Precise definitions, exclusions, conditions, and obligations.
- Formal language from official communications.
Your generated queries must reflect this formal, specific, and structured nature.

# OBJECTIVE
Analyze the user's question and generate a structured set of 3-4 distinct search queries. Each query must be strategically designed to retrieve the most relevant clauses, rules, or statements from the specified document types for use in a hybrid search system.

# INSTRUCTIONS
1.  **Keyword / Jargon Query:** Formulate a query using the precise legal, technical, or contractual jargon likely to be found verbatim in the document. Focus on "terms of art" or specific phrases (e.g., "limitation of liability," "confidentiality clause," "initial waiting period"). This is for a keyword search engine (like BM25).

2.  **Semantic / Intent Query:** Formulate a query that captures the user's underlying intent in a more descriptive way, but maintains a professional tone suitable for a corporate or legal context. This is for a vector search engine and helps find conceptually related clauses that may not use the exact keywords.

3.  **Sub-Question / "Fine Print" Query:** Formulate a query that targets a critical sub-aspect, such as a **waiting period, an exclusion clause, a notice period, a renewal condition, or a specific definition.** This is crucial for finding the controlling "fine print" that a broader query might miss.

4.  **Hypothetical Clause Query:** Formulate a query as a complete, declarative statement that sounds as if it were **pulled directly from the contract or policy.** This powerful technique helps the vector search find the exact clause that confirms or denies the rule.

5.  **Output Format:** You MUST return the queries as a single, well-formed JSON object with the keys: `keyword_query`, `semantic_query`, `fine_print_query`, and `hypothetical_clause_query`. Do not add any other text or explanation.

# USER QUESTION
{question}
"""

class AdvancedRankFusion:
    """Advanced Rank Fusion with multiple retrieval strategies"""
    
    def __init__(self, index_name: str, embedding, llm):
        self.index_name = index_name
        self.embedding = embedding
        self.llm = llm
        self.vector_store = None
        self._setup_vector_store()
        
    def _setup_vector_store(self):
        """Initialize vector store"""
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embedding,
                namespace="default"
            )
        except Exception as e:
            pass
            
    def _generate_diverse_queries(self, original_query: str) -> List[str]:
        """Generate diverse queries using LLM"""
        try:
            structured_llm = self.llm.with_structured_output(Queries)
            
            prompt = QUERY_GENERATION_PROMPT.format(question=original_query)
            result = structured_llm.invoke(prompt)
            
            # Always include the original query
            queries = [original_query] + result.queries
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
                    
            return unique_queries
            
        except Exception as e:
            return [original_query]
    
    def _get_all_document_texts(self) -> List[str]:
        """Retrieve all document texts for BM25 corpus"""
        try:
            # Get all vectors from Pinecone to build BM25 corpus
            index = self.vector_store.index
            
            # Query with empty vector to get all documents
            query_response = index.query(
                vector=[0.0] * 768,  # Zero vector
                top_k=1000,  # Adjust based on your corpus size
                include_metadata=True,
                namespace="default"
            )
            
            texts = []
            for match in query_response.matches:
                if 'text' in match.metadata:
                    texts.append(match.metadata['text'])
                    
            return texts
            
        except Exception as e:
            return []
    
    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Implement Reciprocal Rank Fusion algorithm"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                # Use document content as unique identifier
                doc_id = hash(doc.page_content[:200])  # Hash first 200 chars for uniqueness
                doc_objects[doc_id] = doc
                
                # RRF formula: 1 / (k + rank)
                score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
                doc_scores[doc_id] += score
        
        # Sort by combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return documents in order of fused ranking
        return [doc_objects[doc_id] for doc_id, score in sorted_docs]
    
    def _semantic_reranking(self, documents: List[Document], query: str, top_k: int = 10) -> List[Document]:
        """Additional semantic reranking using query-document similarity"""
        if not documents:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding.embed_query(query)
            
            # Get document embeddings (simplified - in practice, you might want to cache these)
            doc_embeddings = self.embedding.embed_documents([doc.page_content for doc in documents])
            
            # Calculate cosine similarities
            similarities = []
            for doc_emb in doc_embeddings:
                similarity = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                similarities.append(similarity)
            
            # Sort documents by similarity
            doc_sim_pairs = list(zip(documents, similarities))
            doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, sim in doc_sim_pairs[:top_k]]
            
        except Exception as e:
            return documents[:top_k]
    
    def retrieve(self, query: str, top_k: int = 8) -> List[Document]:
        """Main retrieval method with advanced rank fusion"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Step 1: Generate diverse queries
        diverse_queries = self._generate_diverse_queries(query)
        
        # Step 2: Set up retrievers
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k * 2}  # Get more candidates
        )
        
        # Step 3: Vector retrieval for all queries
        all_vector_results = []
        for q in diverse_queries:
            try:
                results = vector_retriever.invoke(q)
                if results:
                    all_vector_results.append(results[:top_k])
            except Exception as e:
                pass
        
        # Step 4: BM25 retrieval (if corpus available)
        bm25_results = []
        try:
            corpus_texts = self._get_all_document_texts()
            if corpus_texts:
                bm25_retriever = BM25Retriever.from_texts(corpus_texts)
                bm25_retriever.k = top_k
                
                for q in diverse_queries:
                    results = bm25_retriever.invoke(q)
                    if results:
                        bm25_results.append(results[:top_k])
        except Exception as e:
            pass
        
        # Step 5: Combine all ranked lists
        all_ranked_lists = all_vector_results + bm25_results
        
        if not all_ranked_lists:
            return []
        
        # Step 6: Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_ranked_lists)
        
        # Step 7: Semantic reranking
        final_results = self._semantic_reranking(fused_results, query, top_k)
        
        return final_results

def get_rank_fusion_retriever(index_name: str, embedding, query: str, llm):
    """Factory function to create advanced rank fusion retriever"""
    fusion = AdvancedRankFusion(index_name, embedding, llm)
    
    class RankFusionRetriever:
        def __init__(self, fusion_instance):
            self.fusion = fusion_instance
            
        def invoke(self, query: str) -> List[Document]:
            return self.fusion.retrieve(query)
    
    return RankFusionRetriever(fusion)