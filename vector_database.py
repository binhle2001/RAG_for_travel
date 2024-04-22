from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings


loader = PyPDFDirectoryLoader("TaiLieuDulich1")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
documents = text_splitter.split_documents(docs)



# Load the embedding model 
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)



DB_URL = "http://localhost:6333"
COLLECTION = "RAG_CHATBOT"

vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=DB_URL,
    collection_name=COLLECTION,
)

# Initialize the vector retriever
# bm25_retriever = BM25Retriever.from_documents(documents)
# bm25_retriever.k = 3  