from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from text_extraction.DAL_files.trademark_dal import TrademarkExtractor
from text_extraction.DAL_files.trademark_comparison import TrademarkComparator
from text_extraction.DAL_files.csv_manager import TrademarkCSVManager
import tempfile
import re
import base64

from dotenv import load_dotenv
load_dotenv()

# For demo purposes, we'll use a hardcoded  key or environment variable
# In production, you should get this from a secure configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

trademark_router = APIRouter()
trademark_extractor = TrademarkExtractor(groq_api_key=GROQ_API_KEY)
trademark_comparator = TrademarkComparator()
csv_manager = TrademarkCSVManager()

"""
Trademark extraction endpoints for extracting trademark data from text, PDF, or images.
"""


@trademark_router.post("/extract/trademark/image")
async def extract_trademark_from_image(file: UploadFile = File(...)):
    """
    Extract trademark data from an uploaded image file (JPG, PNG, etc.).
    """
    try:
        # Validate file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content and convert to base64
        file_bytes = await file.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        
        # Extract text from image
        extracted_text = trademark_extractor.extract_from_base64_image(base64_image)
        
        if not extracted_text or not extracted_text.get("text"):
            raise HTTPException(status_code=400, detail="Could not extract text from image")
        
        # Extract structured trademark data
        trademark_data = trademark_extractor.extract_trademark_data_from_text(extracted_text["text"])
        
        return {
            "filename": file.filename,
            "extracted_text": extracted_text["text"],
            "trademark_data": trademark_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@trademark_router.post("/extract/trademark/base64")
async def extract_trademark_from_base64(request: dict):
    """
    Extract trademark data from a base64 encoded image.
    Expected input: {"image": "base64-encoded-image-data"}
    """
    try:
        base64_image = request.get("image", "")
        if not base64_image:
            raise HTTPException(status_code=400, detail="Base64 image data is required")
        
        # Remove data URL prefix if present
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Extract text from image
        extracted_text = trademark_extractor.extract_from_base64_image(base64_image)
        
        if not extracted_text or not extracted_text.get("text"):
            raise HTTPException(status_code=400, detail="Could not extract text from image")
        
        # Extract structured trademark data
        trademark_data = trademark_extractor.extract_trademark_data_from_text(extracted_text["text"])
        
        return {
            "extracted_text": extracted_text["text"],
            "trademark_data": trademark_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@trademark_router.get("/trademark/health")
async def health_check():
    """
    Health check endpoint to verify the trademark extraction service is running.
    """
    return {
        "status": "healthy",
        "service": "trademark_extraction_and_comparison",
        "version": "1.0.0",
        "csv_file": csv_manager.csv_file_path
    } 
