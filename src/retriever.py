import warnings
import logging
import os
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "C:/Users/Thabasvini/OneDrive/Desktop/QA bot/chroma_store"
TOP_K = 4

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt = PromptTemplate.from_template("""
Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't have enough information in my documents to answer this."

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question: str):
    answer = chain.invoke(question)
    print("\n📌 Answer:", answer)
    docs = retriever.invoke(question)
    print("\n📄 Sources:")
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        entry = f"  - {src} | page {page}"
        if entry not in seen:
            print(entry)
            seen.add(entry)

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        ask(q)