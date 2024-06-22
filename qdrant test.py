from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_qdrant import Qdrant

import time
import os
from dotenv import load_dotenv
_ = load_dotenv(dotenv_path=".env")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

ollama_emb = OllamaEmbeddings(
    model=os.environ["OLLAMA_EMBEDDING_MODEL"],
    model_kwargs={"keep_alive": 1},
)

with open("temp.txt", "r", encoding="utf-8") as f:
    part = f.read()  # str
    chunks = text_splitter.create_documents([part])  # Chunks ready for RAG | List of Documents

# qdrant loading time
start = time.time()
# qdrant = Qdrant.from_documents(
#     documents=chunks,
#     embedding=ollama_emb,
#     path="./embeddings/Qdrant",
#     collection_name="CoFi",
# )
qdrant = Qdrant.from_existing_collection(
    path="./embeddings/Qdrant",
    collection_name="CoFi",
    embedding=ollama_emb,
)
t = time.time() - start
print(f"Qdrant loading time: {t:.2f} seconds")


query = ""
retriever = qdrant.as_retriever()

docs = retriever.invoke("JEBA REZWANA and MARY LOU MAHER")

print([doc.page_content for doc in docs])