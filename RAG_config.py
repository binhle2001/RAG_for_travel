from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


DB_URL = "http://localhost:6333"
COLLECTION = "RAG_CHATBOT"


model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

client = QdrantClient(
    url=DB_URL, prefer_grpc=False
)

qdrant = Qdrant(client=client, embeddings=embeddings, collection_name=COLLECTION)
qdrant_retriever = qdrant.as_retriever(search_kwargs={"k": 3})

loader = PyPDFDirectoryLoader("TaiLieuDulich1")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
documents = text_splitter.split_documents(docs)

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3 

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, qdrant_retriever], weights=[0.4, 0.6]
)