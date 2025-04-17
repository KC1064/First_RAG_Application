from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import google.generativeai as genai

genai.configure(api_key="")

# Load the PDF document
pdf_path = Path(__file__).parent / "sample.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
split_docs = text_splitter.split_documents(documents=docs)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=""  # Notice parameter name change
)
# Initialize Qdrant client explicitly
client = QdrantClient(url="http://localhost:6333")

# Create or retrieve collection
collection_name = "learning_langchain"  # Changed name to remove space
# try:
#     # Create the vector store with documents
#     vector_store = QdrantVectorStore.from_documents(
#         documents=split_docs,
#         embedding=embeddings,
#         url="http://localhost:6333",
#         collection_name=collection_name,
#         force_recreate=True  # Set to False if you want to reuse an existing collection
#     )
#     print("Documents successfully added to vector store")
# except Exception as e:
#     print(f"Error creating vector store: {e}")

retriver = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=collection_name,
)

search_result = retriver.similarity_search(
    query = "What is lorem"
)

print(search_result)