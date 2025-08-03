import os
from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import Pinecone as PineconeVectorStore # ← Updated import

load_dotenv()

llm         = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embedding   = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index_name  = os.getenv("PINECONE_INDEX_NAME") or "aurag"

def query_document(query: str):
    # This LangChain function now correctly initializes the new Pinecone client
    # using environment variables under the hood.
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="default"
    )
    docs  = vectorstore.similarity_search(query, k=5)
    chain = load_qa_chain(llm, chain_type="stuff")
    # chain.run is deprecated, prefer .invoke for new LangChain versions
    return chain.invoke({"input_documents": docs, "question": query})['output_text']