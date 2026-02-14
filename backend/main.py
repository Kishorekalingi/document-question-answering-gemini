from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import uuid
import os
import shutil
from document_processor import DocumentProcessor
from gemini_client import GeminiClient
from session_manager import SessionManager
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
processor = DocumentProcessor()
gemini = GeminiClient(api_key=os.getenv("GEMINI_API_KEY", ""))
session_manager = SessionManager(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None)
)

# FastAPI app
app = FastAPI(title="Document Question Answering System")

# In-memory document store (for demo purposes)
documents: Dict[str, Dict] = {}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document"""
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file temporarily
    temp_dir = f"./temp_{document_id}"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store document info
    documents[document_id] = {
        "document_id": document_id,
        "filename": file.filename,
        "status": "processing",
        "file_path": file_path,
        "chunks": None
    }
    
    # Process document in background
    # In production, use Celery or similar for background tasks
    process_document(document_id)
    
    return {
        "document_id": document_id,
        "filename": file.filename,
        "message": "Document accepted for processing."
    }

def process_document(document_id: str):
    """Process document in background"""
    try:
        doc_info = documents[document_id]
        file_path = doc_info["file_path"]
        
        # Extract text
        text = processor.extract_text(file_path)
        
        # Chunk text
        chunks = processor.chunk_text(text)
        
        # Store chunks
        documents[document_id]["chunks"] = chunks
        documents[document_id]["status"] = "completed"
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        documents[document_id]["status"] = "failed"

@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "status": documents[document_id]["status"]
    }

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Get document text chunks"""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if documents[document_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Document not yet processed")
    
    return {
        "document_id": document_id,
        "chunks": documents[document_id]["chunks"]
    }

@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    """Ask a question about uploaded documents"""
    session_id = question_request.session_id
    document_ids = question_request.document_ids
    question = question_request.question
    
    # Validate documents
    for doc_id in document_ids:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        if documents[doc_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Document {doc_id} not yet processed")
    
    # Get all chunks from requested documents
    all_chunks = []
    for doc_id in document_ids:
        all_chunks.extend(documents[doc_id]["chunks"])
    
    # Generate answer using Gemini API
    result = gemini.generate_answer_batched(question, all_chunks, batch_size=5)
    
    # Add to conversation history
    session_manager.add_message(session_id, "user", question)
    session_manager.add_message(session_id, "assistant", result["answer"])
    
    return AnswerResponse(
        answer=result["answer"],
        session_id=session_id,
        source_chunks=[{"document_id": doc_id, "chunk_id": chunk["chunk_id"], "text": chunk["text"]} 
                       for chunk in all_chunks],
        batch_size=result["batch_size"],
        tokens_used=TokenUsage(**result["tokens_used"])
    )

@app.get("/session/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history"""
    history = session_manager.get_history(session_id)
    return ConversationHistory(
        session_id=session_id,
        history=history
    )

@app.get("/session/{session_id}/export")
async def export_conversation(session_id: str):
    """Export conversation as PDF"""
    from fpdf import FPDF
    
    history = session_manager.get_history(session_id)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Conversation History", 0, 1, "C")
    pdf.ln(10)
    
    for message in history:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{message['role'].capitalize()}: ", 0, 0)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, message["content"])
        pdf.ln(4)
    
    # Generate PDF content
    pdf_content = pdf.output(dest="S").encode("latin-1")
    
    return StreamingResponse(
        iter([pdf_content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.pdf"}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Document Question Answering System - Ready to use!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
