from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from src.search import RAGSearch
from pathlib import Path
from llm.brochure import create_brochure as generate_brochure

from fastapi import FastAPI, UploadFile, File, HTTPException

from ingest import ingest_documents

from src.data_loader import load_single_file
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag_search = RAGSearch()


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Knowledge Assistant API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    result = rag_search.search_and_summarize(
        request.question,
        top_k=3
    )

    return result

    
# Folder where uploaded files will be saved
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File types allowed by our document loader
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".xlsx",
    ".docx",
    ".json",
}

# upload file endpoint

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Check that a filename was provided
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    # 2. Extract the filename safely
    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    # 3. Check the file extension
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}"
        )

    # 4. Create the destination path
    file_path = UPLOAD_DIR / filename

    # 5. Prevent accidentally overwriting an existing file
    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail="A file with this name already exists."
        )

    try:
        # 6. Save the uploaded file
        contents = await file.read()

        # Basic file-size limit: 20 MB
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 20 MB."
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        file_path.write_bytes(contents)

        # 7. Rebuild the vector store using all documents in data/
        documents = load_single_file(str(file_path))

        # 8. Reload the new FAISS index into the current RAG search object
        rag_search.vectorstore.add_documents(documents)

        # 9. Return a success response
        return {
            "message": "File uploaded and indexed successfully.",
            "filename": filename,
            "file_type": extension,
            "size_bytes": len(contents)
        }

    except HTTPException:
        # Preserve intentional HTTP errors
        if file_path.exists():
            file_path.unlink()
        raise

    except Exception as e:
        # Remove the uploaded file if processing fails
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )

    finally:
        await file.close()



class BrochureRequest(BaseModel):
    company_name: str
    url: HttpUrl


@app.post("/brochure")
def create_brochure(request: BrochureRequest):
    brochure = generate_brochure(
        company_name=request.company_name,
        url=str(request.url)
    )

    return {
        "company_name": request.company_name,
        "url": str(request.url),
        "brochure": brochure
    }

