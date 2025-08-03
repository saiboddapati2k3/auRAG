import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()

# Initialize LLM and embedding
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def query_document(query: str):
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # No need to create or manage Pinecone.Index manually — langchain handles it
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="default"
    )

    # Perform vector similarity search
    docs = vectorstore.similarity_search(query, k=5)

    # Generate answer using GPT-4
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    return answer
