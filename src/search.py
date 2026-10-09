import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        # since dont need to build the vector store while calling an API we commented it out
        # faiss_path = os.path.join(persist_dir, "faiss.index")
        # meta_path = os.path.join(persist_dir, "metadata.pkl")

        # if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        #     from src.data_loader import load_all_documents
        #     docs = load_all_documents("data")
        #     self.vectorstore.build_from_documents(docs)
        # else:
        #     self.vectorstore.load()

        self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""Summarize the following context for the query: '{query}'

Context:
{context}

Summary:"""

        response = self.llm.invoke([prompt])

        sources = []

        for result in results:
            if result["metadata"]:
                sources.append({
                        "text": result["metadata"].get("text", ""),
                        "source": result["metadata"].get("source", ""),
                        "page": result["metadata"].get("page", None),
                        "distance": float(result["distance"])
                })

        return {
            "answer": response.content,
            "sources": sources
        }