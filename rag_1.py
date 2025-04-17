#!/usr/bin/env python3
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
import google.generativeai as genai
import os
import sys
import argparse
import time
import textwrap

# --- Config ---
API_KEY = ""

# --- CLI Colors and Formatting ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_response(text):
    # Format the response for better readability
    wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
    paragraphs = text.split('\n')
    formatted_text = "\n".join(["\n".join(wrapper.wrap(p)) if p else "" for p in paragraphs])
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}💬 Answer:{Colors.ENDC}")
    print(f"{Colors.GREEN}{'-'*40}{Colors.ENDC}")
    print(formatted_text)
    print(f"{Colors.GREEN}{'-'*40}{Colors.ENDC}\n")

def loading_animation(task):
    """Show a simple loading animation"""
    animation = "|/-\\"
    idx = 0
    print(f"\r{task}", end="")
    for _ in range(10):
        print(f"\r{task} {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.1)
    print("\r" + " " * (len(task) + 2), end="")
    print("\r", end="")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="PDF-based Chat CLI System")
    parser.add_argument("--pdf", default="demo.pdf", help="Path to the PDF file")
    parser.add_argument("--collection", default="Software_Engg", help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Force recreate the vector store")
    parser.add_argument("--chunk-size", type=int, default=100, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Text chunk overlap")
    args = parser.parse_args()

    try:
        # Configure Gemini API
        genai.configure(api_key=API_KEY)
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header("📘 SOFTWARE ENGINEERING PDF CHAT SYSTEM")
        print()
        
        # --- Load PDF and preprocess ---
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print_error(f"PDF file not found at {pdf_path}")
            return
            
        print_info(f"Loading PDF from {pdf_path}...")
        loading_animation("Processing PDF")
        
        try:
            loader = PyPDFLoader(file_path=str(pdf_path))
            docs = loader.load()
            print_success(f"Loaded {len(docs)} pages from PDF")
        except Exception as e:
            print_error(f"Failed to load PDF: {e}")
            return

        # --- Split docs ---
        print_info("Splitting text into chunks...")
        loading_animation("Splitting text")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents=docs)
        print_success(f"Split into {len(split_docs)} chunks")

        # --- Embeddings ---
        print_info("Initializing embedding model...")
        loading_animation("Loading model")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )

        # --- Connect to Qdrant ---
        collection_name = args.collection
        try:
            print_info(f"Connecting to Qdrant at http://localhost:6333...")
            loading_animation("Connecting")
            client = QdrantClient(url="http://localhost:6333")
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            # Populate vector DB if needed
            if not collection_exists or args.recreate:
                print_info(f"Creating collection '{collection_name}'...")
                loading_animation("Creating vectors")
                vector_store = QdrantVectorStore.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    url="http://localhost:6333",
                    collection_name=collection_name,
                    force_recreate=True
                )
                print_success("Vector store populated successfully")
            else:
                print_success(f"Using existing collection '{collection_name}'")
                
        except qdrant_exceptions.UnexpectedResponse as e:
            print_error(f"Qdrant server error: {e}")
            print_warning("Make sure Qdrant server is running at http://localhost:6333")
            return
        except Exception as e:
            print_error(f"Error connecting to Qdrant: {e}")
            return

        # --- Create vector store for retrieval ---
        # Fix: Use correct initialization parameters for QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings  # Changed from 'embeddings' to 'embedding'
        )
        
        # Create retriever from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # --- Initialize LLM ---
        print_info("Initializing Gemini language model...")
        loading_animation("Setting up")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=API_KEY,
            temperature=0.3
        )

        system_prompt = """
        You are a helpful AI assistant designed to answer technical queries based on 
        software engineering documents. Give clear, accurate, and well-formatted responses 
        in plain CLI-friendly text. Keep answers concise but informative. 
        Follow these rules:
        1. If there are spelling mistake then correct it and search for that.
        2. Don't give uneven or half answer use all the information and give answer precisely.
        3. If you don't find the user query's response suggest the page no. where they can find that 
        """

        print()
        print_header("CHAT MODE INITIALIZED")
        print_info("Type your questions about the document below")
        print_warning("Type 'exit' or 'quit' to end the session")
        print()

        # Chat history for context
        chat_history = []

        while True:
            try:
                query = input(f"{Colors.BLUE}{Colors.BOLD}🔍 Question: {Colors.ENDC}")
                if query.lower() in ['exit', 'quit']:
                    print_success("Exiting. Have a great day!")
                    break
                
                if not query.strip():
                    continue
                    
                loading_animation("Searching document")
                
                # Step 1: Retrieve relevant chunks
                relevant_docs = retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                
                loading_animation("Generating response")
                
                # Step 2: Ask LLM
                full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nChat History:\n{chat_history}\n\nUser Question: {query}\n\nAnswer:"
                response = llm.invoke(full_prompt)
                
                # Update chat history (keep last 3 Q&A pairs)
                chat_history.append(f"Q: {query}\nA: {response.content}")
                if len(chat_history) > 3:
                    chat_history = chat_history[-3:]
                
                # Step 3: Display response
                print_response(response.content)
                
            except KeyboardInterrupt:
                print()
                print_warning("Interrupted by user. Exiting...")
                break
            except Exception as e:
                print_error(f"Error: {e}")
                print_error(f"Error details: {str(e.__class__.__name__)}")

    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        print_error(f"Error type: {str(e.__class__.__name__)}")

if __name__ == "__main__":
    main()