import warnings
import logging
import os
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PERSIST_DIR = "C:/Users/Thabasvini/OneDrive/Desktop/QA bot/chroma_store"
TOP_K = 4

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Document Q&A Bot")
st.markdown("Ask questions from your document knowledge base.")

@st.cache_resource
def load_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)

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

    return chain, retriever

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(src)

if question := st.chat_input("Ask a question from your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = load_chain()
            answer = chain.invoke(question)
            docs = retriever.invoke(question)

            sources = []
            seen = set()
            for doc in docs:
                src = doc.metadata.get("source", "unknown").split("\\")[-1]
                page = doc.metadata.get("page", "?")
                entry = f"📄 `{src}` — page {page}"
                if entry not in seen:
                    sources.append(entry)
                    seen.add(entry)

            st.markdown(answer)
            with st.expander("📄 Sources"):
                for src in sources:
                    st.markdown(src)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })