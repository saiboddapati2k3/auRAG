import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from rag_fusion import get_rank_fusion_retriever

load_dotenv()

# Enhanced QA prompt
QA_PROMPT_TEMPLATE = """
# ROLE
You are a meticulous and impartial Analyst. Your primary function is to provide precise, factual answers by synthesizing information exclusively from the provided text. You are not a conversational chatbot; you are a fact-based analytical engine.

# CORE DIRECTIVE
Your response MUST be based 100% on the `Provided Context`. You are strictly forbidden from using any prior knowledge or external information. Every part of your answer must be directly supported by the text provided.

---
# Provided Context:
{context}
---
# User's Question:
{question}
---

# INSTRUCTIONS FOR YOUR RESPONSE:

1.  **Reason First, Then Answer:** Before writing, perform a step-by-step analysis to connect the specific rules and statements in the `Provided Context` to the `User's Question`.

2.  **Cite Your Sources:** This is your most important task. For every piece of information you provide in your answer, you MUST reference the specific source clause or document part it came from. Assume the context chunks have metadata IDs (e.g., [SEC-3.1.b], [policy_doc_p4]). Use this format: `The waiting period for specific surgeries is 24 months [SEC-3.4.b].`

3.  **Answer Directly and Concisely:** Begin your response with a direct summary answering the user's question.

4.  **Elaborate with Details:** After the summary, provide a more detailed breakdown. Use bullet points or numbered lists to present specific conditions, amounts, timelines, and exclusions mentioned in the context. Always include citations.

5.  **Handle Insufficient Information:** If the context does not contain the information to answer the question, state that directly. Do not speculate or apologize. Clearly specify what information is missing. Example: `The provided context does not specify the procedure for out-of-network claims.`

6.  **Maintain a Formal Tone:** Structure your response clearly and professionally. Avoid conversational filler, opinions, or any language that is not directly supported by the context.

# Answer:
"""

class OptimizedQAModel:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Slightly higher for more natural responses
            # max_tokens=2048
        )
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "aurag")
        
        # Custom prompt
        self.qa_prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Load QA chain with custom prompt
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=self.qa_prompt
        )
    
    def query_document(self, query: str) -> str:
        """Enhanced document querying with rank fusion"""
        try:
            # Get retriever with rank fusion
            retriever = get_rank_fusion_retriever(
                self.index_name, 
                self.embedding, 
                query, 
                self.llm
            )
            
            # Retrieve relevant documents using invoke
            docs = retriever.invoke(query)
            
            if not docs:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Prepare context - limit size to avoid token limits
            context_parts = []
            total_chars = 0
            max_context_chars = 8000  # Limit context size
            
            for i, doc in enumerate(docs):
                doc_content = doc.page_content.strip()
                if total_chars + len(doc_content) > max_context_chars:
                    break
                context_parts.append(f"Document {i+1}:\n{doc_content}")
                total_chars += len(doc_content)
            
            # Generate answer using the QA chain with timeout handling
            try:
                result = self.qa_chain.invoke({
                    "input_documents": docs[:len(context_parts)],  # Only use docs that fit in context
                    "question": query
                })
                
                answer = result.get('output_text', 'Sorry, I could not generate an answer.')
                
                return answer
                
            except Exception as llm_error:
                # Fallback: provide a simple context-based response
                fallback = f"LLM failed, but based on retrieved documents: {context_parts[0][:200] if context_parts else 'No context available'}"
                return fallback
            
        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"

# Global instance
qa_model = OptimizedQAModel()
query_document = qa_model.query_document
