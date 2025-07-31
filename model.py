import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

# Setup LLM and embedding model
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()

def query_document(query: str):
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Load vector store (must be pre-populated)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="default"
    )

    # Search relevant documents
    docs = vectorstore.similarity_search(query, k=5)

    # Answer with LLM
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    return answer
