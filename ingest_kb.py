import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def setup_mock_data(kb_dir="./kb_data"):
    """
    Checks if the KB directory exists and has files.
    If not, creates it and generates mock customer support articles.
    """
    os.makedirs(kb_dir, exist_ok=True)
    
    existing_files = glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md"))
    
    if not existing_files:
        print(f"No documents found in {kb_dir}. Generating mock data...")
        mock_data = {
            "refund_policy.txt": (
                "Refund Policy\n\n"
                "Customers can request a refund within 30 days of purchase. "
                "The item must be in its original packaging and unused. "
                "Refunds are processed to the original form of payment within 5-7 business days."
            ),
            "password_reset.txt": (
                "How to Reset Your Password\n\n"
                "If you forgot your password, go to the login page and click 'Forgot Password'. "
                "Enter your registered email address, and we will send you a reset link. "
                "Click the link and follow the instructions to create a new password."
            ),
            "shipping_delays.txt": (
                "Shipping and Delivery Delays\n\n"
                "Standard shipping usually takes 3-5 business days. "
                "However, due to high volume, some shipments may experience a delay of 1-2 days. "
                "You can track your order using the tracking link in your confirmation email."
            )
        }
        
        for filename, content in mock_data.items():
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        print("Mock data generated successfully.")
    else:
        print(f"Found {len(existing_files)} documents in {kb_dir}.")

def load_and_split_documents(kb_dir="./kb_data", chunk_size=500, chunk_overlap=50):
    """
    Reads all .txt and .md files from the KB directory and splits them into chunks.
    """
    print("Loading and splitting documents...")
    files = glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.md"))
    
    # Initialize the LangChain text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    metadatas = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        file_chunks = splitter.split_text(text)
        for chunk in file_chunks:
            chunks.append(chunk)
            metadatas.append({"source": filename})
            
    print(f"Created {len(chunks)} chunks from {len(files)} files.")
    return chunks, metadatas

def initialize_vector_db(chunks, metadatas, db_dir="./models/vector_db"):
    """
    Embeds the chunked text and stores it in a persistent ChromaDB collection.
    """
    print("Initializing embedding model and vector database...")
    # Initialize the lightning-fast sentence transformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_dir)
    
    # Get or create the collection
    collection = chroma_client.get_or_create_collection(name="support_kb")
    
    print("Embedding chunks and adding to the database...")
    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Since ChromaDB requires unique IDs for each document
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Database successfully built and persisted.")
    return collection, embedding_model

def retrieve_test(collection, embedding_model, query="How do I reset my password?", top_k=2):
    """
    Test routine to verify retrieval functionality.
    """
    print(f"\n--- Testing Retrieval ---")
    print(f"Query: '{query}'")
    
    # Embed the query to match against DB embeddings
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Query the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # Display results
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i in range(len(retrieved_docs)):
        print(f"\nResult {i+1} (Distance: {distances[i]:.4f}):")
        print(f"Source: {retrieved_metadatas[i]['source']}")
        print(f"Text: {retrieved_docs[i]}")
        
    print("\nRetrieval test completed successfully.")
    
if __name__ == "__main__":
    # 1. Setup Mock Data
    setup_mock_data()
    
    # 2. Document Parsing & Splitting
    chunks, metadatas = load_and_split_documents()
    
    # 3. Embeddings & Vector DB Persistence
    if chunks:
        collection, embedding_model = initialize_vector_db(chunks, metadatas)
        
        # 4. Verification & Test Retrieval
        retrieve_test(collection, embedding_model, query="How do I reset my password?", top_k=2)
    else:
        print("No chunks to process. Exiting.")
