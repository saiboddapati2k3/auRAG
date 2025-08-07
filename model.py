import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from rag_fusion import get_rank_fusion_retriever

load_dotenv()

# Enhanced QA prompt
QA_PROMPT_TEMPLATE = """
You are an expert assistant that provides accurate, comprehensive answers based on the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, clearly state what's missing
3. Provide specific details and examples when available
4. Structure your response clearly with relevant sections
5. If multiple perspectives exist in the context, present them fairly

Answer:
"""

class OptimizedQAModel:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Slightly higher for more natural responses
            max_tokens=2048
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
            print(f"🤖 Processing query: {query}")
            
            # Get retriever with rank fusion
            retriever = get_rank_fusion_retriever(
                self.index_name, 
                self.embedding, 
                query, 
                self.llm
            )
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "I couldn't find relevant information in the document to answer your question."
            
            print(f"📚 Found {len(docs)} relevant documents")
            
            # Generate answer using the QA chain
            result = self.qa_chain.invoke({
                "input_documents": docs,
                "question": query
            })
            
            return result.get('output_text', 'Sorry, I could not generate an answer.')
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return f"An error occurred while processing your question: {str(e)}"

# Global instance
qa_model = OptimizedQAModel()
query_document = qa_model.query_document