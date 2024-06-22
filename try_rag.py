#  RAG pipeline trial
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import Qdrant
 

import os
from dotenv import load_dotenv

_ = load_dotenv(dotenv_path=".env")

os.environ["PATH"] = (
    os.environ["PATH"]
    + ";"
    + os.environ["POPPLER_PATH"]
    + ";"
    + os.environ["TESSERACT_PATH"]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

if not os.path.isfile("temp.txt"):
    with open("temp.txt", "w", encoding="utf-8") as f:
        loader = UnstructuredFileLoader(
            "./Docs/input/Designing Creative AI Partners with COFI A Framework for Modeling Interaction in Human-AI Co-Creative Systems.pdf",
            # mode="elements",
            strategy="hi_res",
            post_process=[clean_extra_whitespace],
        )
        part = loader.load()  # list of Document
        f.write(part[0].page_content)
        chunks = text_splitter.create_documents([part[0].page_content])  # Chunks ready for RAG | List of Documents
else:
    with open("temp.txt", "r", encoding="utf-8") as f:
        part = f.read()  # str
        chunks = text_splitter.create_documents([part])  # Chunks ready for RAG | List of Documents

ollama_emb = OllamaEmbeddings(
    model=os.environ["OLLAMA_EMBEDDING_MODEL"],
    model_kwargs={"keep_alive": 1},
)

# embeddings = ollama_emb.embed_documents(chunks[:3])

store = LocalFileStore("./embeddings/Qdrant")

cache_backed_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=ollama_emb,
    document_embedding_cache=store,
    namespace=ollama_emb.model,
)
# cache = cache_backed_embeddings.embed_documents(chunks[:3]) # Failure needs str idk why whereas ollama_emb.embed_documents() works fine
print("Making cache")
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=ollama_emb,
    path="./embeddings/Qdrant",
    collection_name="CoFi",
)
print("Cache made")

query = "Author of the COFI paper"
found_docs = qdrant.similarity_search(query)

for i in found_docs:
    print(i.page_content)