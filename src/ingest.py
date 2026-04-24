import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time
import shutil

load_dotenv()

DATA_PATH = "C:/Users/Thabasvini/OneDrive/Desktop/QA bot/data"
PERSIST_DIR = "C:/Users/Thabasvini/OneDrive/Desktop/QA bot/chroma_store"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print("🗑️ Old chroma store deleted")

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        file_path = os.path.join(DATA_PATH, file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        print(f"  Loaded: {file}")

print(f"\n✅ Loaded {len(documents)} pages total")

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

#Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

print("⏳ Embedding and storing in Chroma in batches...")

BATCH_SIZE = 80  # slightly under 100/min limit

db = None
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"  Processing batch {batch_num} / {total_batches} ({len(batch)} chunks)...")

    if db is None:
        db = Chroma.from_documents(
            batch,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    else:
        db.add_documents(batch)

    db.persist()  # Save after every batch

    if i + BATCH_SIZE < len(chunks):
        print("  ⏳ Waiting 65 seconds for rate limit...")
        time.sleep(65)

print(f"✅ Documents indexed successfully! Total: {db._collection.count()} chunks")