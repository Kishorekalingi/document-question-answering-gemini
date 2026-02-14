from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Document(BaseModel):
    document_id: str
    filename: str

class DocumentStatus(BaseModel):
    document_id: str
    status: str  # 'processing', 'completed', 'failed'

class TextChunk(BaseModel):
    chunk_id: int
    text: str

class DocumentChunks(BaseModel):
    document_id: str
    chunks: List[TextChunk]

class QuestionRequest(BaseModel):
    session_id: str
    document_ids: List[str]
    question: str

class SourceChunk(BaseModel):
    document_id: str
    chunk_id: int
    text: str

class TokenUsage(BaseModel):
    prompt_tokens: int
    candidates_tokens: int
    total_tokens: int

class AnswerResponse(BaseModel):
    answer: str
    session_id: str
    source_chunks: List[SourceChunk]
    batch_size: int
    tokens_used: TokenUsage

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ConversationHistory(BaseModel):
    session_id: str
    history: List[Message]

class ExportRequest(BaseModel):
    session_id: str
