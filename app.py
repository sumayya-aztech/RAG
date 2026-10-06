from src.data_loader import load_all_documents
# from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
     #you dont need to run it everytime unless you dont have a new document
    #docs = load_all_documents("data")


    #test document.py
    # print(docs)
    # print(f"Loaded {len(docs)} documents.")
    # print("Example document:", docs[0] if docs else None)


    #test embedding.py
    # chunks = EmbeddingPipeline().chunk_documents(docs)
    # chunkvectors = EmbeddingPipeline().embed_chunks(chunks)
    # print(chunkvectors)

    #you dont need to run it everytime unless you dont have a new document
     #docs = load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("Explain artificial intelligence and its applications?, top_k=3 "))

    rag_search = RAGSearch()
    query = "what is the Knowledge-Based Systems(KBS)?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)






