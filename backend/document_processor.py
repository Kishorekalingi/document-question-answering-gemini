import PyPDF2
from docx import Document as DocxDocument
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 100
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on extension"""
        try:
            if file_path.endswith('.pdf'):
                return self._extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self._extract_text_from_docx(file_path)
            elif file_path.endswith('.txt'):
                return self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str) -> List[dict]:
        """Split text into chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # If we're not at the end, try to find a good breaking point
            if end < text_length:
                # Look for sentence or paragraph break
                break_point = text.rfind('.', start, end)
                if break_point == -1:
                    break_point = text.rfind('\n', start, end)
                if break_point != -1 and break_point > start + 50:
                    end = break_point + 1
            
            chunk_text = text[start:end]
            chunks.append({
                'chunk_id': len(chunks),
                'text': chunk_text.strip()
            })
            
            # Move start forward with overlap
            start = end - self.chunk_overlap if end < text_length else end
        
        return chunks
