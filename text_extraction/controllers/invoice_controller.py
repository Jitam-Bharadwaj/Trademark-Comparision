from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
from DAL_files.invoice_dal import SimpleInvoiceExtractor
from schemas.invoice_schemas import InvoiceTextRequest
import tempfile
import re
from config import settings
from dependencies import invoice_usage_checker
from DAL_files.api_usage_dal import ApiUsageDAL
from database import get_session
from sqlmodel.ext.asyncio.session import AsyncSession

from dotenv import load_dotenv
load_dotenv()


invoice_router = APIRouter()
api_key = settings.groq_api_key
invoice_extractor = SimpleInvoiceExtractor(groq_api_key=api_key)
usage_service = ApiUsageDAL()

"""
Invoice extraction endpoints for classifying documents and extracting invoice data from text, PDF, or images.
"""

@invoice_router.post("/extract/invoice")
async def extract_invoice(
    request: InvoiceTextRequest,
    user_id: str = Depends(invoice_usage_checker),
    session: AsyncSession = Depends(get_session)
):
    """
    Classify document type and extract invoice data from provided text.
    """
    try:
        doc_type = invoice_extractor.classify_document(request.text)
        print("-----------------",doc_type,"-------------")
        invoice_data = invoice_extractor.extract_invoice_fromate_from_text(request.text, doc_type)
        
        # Increment invoice usage counter after successful extraction
        await usage_service.increment_invoice_usage(user_id, session)
        
        return invoice_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@invoice_router.post("/extract/pdf-image-text")
async def extract_pdf_image_text(
    file: UploadFile = File(...),
    user_id: str = Depends(invoice_usage_checker),
    session: AsyncSession = Depends(get_session)
):
    """
    Extract invoice data from an uploaded PDF or image file (in-memory, no temp file).
    """
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        file_bytes = file.file.read()
       
        if suffix == ".pdf":
            invoice_data = invoice_extractor.extract_from_pdf_bytes(file_bytes)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            import base64
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            invoice_data = invoice_extractor.extract_from_base64_image(base64_image)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        
        # Increment invoice usage counter after successful extraction
        await usage_service.increment_invoice_usage(user_id, session)
        
        return invoice_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 