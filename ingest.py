from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore


def ingest_documents():
    print("[INFO] Starting document ingestion...")

    # Load documents from data folder
    documents = load_all_documents("data")

    if not documents:
        print("[WARNING] No documents found in data folder.")
        return

    print(f"[INFO] Loaded {len(documents)} documents.")

    # Create vector store
    vectorstore = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Build embeddings and FAISS index
    vectorstore.build_from_documents(documents)

    print("[INFO] Document ingestion completed successfully.")


if __name__ == "__main__":
    ingest_documents()