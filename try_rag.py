#  RAG pipeline trial
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_qdrant import Qdrant
# from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(
    "./Docs/input/Constitution of India.pdf", mode="elements", strategy="hi-res"
)
part = loader.load()

print(type(part))
print(part[:10])