import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()

llm         = ChatOpenAI(model="gpt-4", temperature=0)
embedding   = OpenAIEmbeddings()
index_name  = os.getenv("PINECONE_INDEX_NAME") or "aurag"

def query_document(query: str):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="default"
    )
    docs  = vectorstore.similarity_search(query, k=5)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)
