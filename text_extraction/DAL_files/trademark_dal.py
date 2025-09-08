import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import fitz
import base64
import json
import re
from fastapi import HTTPException
from pydantic import SecretStr

class TrademarkExtractor:
    def __init__(self, groq_api_key: str):
        self.model = ChatGroq(
            temperature=0.1,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=SecretStr(groq_api_key),
        )
        self.model2 = ChatGroq(
            temperature=0.1,
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(groq_api_key),
        )

        self.text_extract_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Extract all text content from this trademark document exactly as it appears.\n      Maintain the original layout and formatting as much as possible.\n      Pay special attention to:\n      1. Company/applicant names and addresses\n      2. Trademark names and logos\n      3. Registration dates and numbers\n      4. Business categories and descriptions\n      5. Contact information and legal details"),

            ("human",
                [
                    {"type":"text", "text": "Extract all text from this trademark document with high accuracy:"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}}
                ])
        ])

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert trademark data extraction assistant. You must return ONLY a valid JSON object with no additional text, formatting, or explanations.

CRITICAL RULES:
1. Return ONLY the JSON object - no markdown, no code blocks, no explanations
2. Ensure all JSON syntax is correct - no trailing commas, proper quotes, valid structure
3. Use consistent field names throughout
4. All string values must be properly quoted
5. Do not include any comments or extra text

Extract the following trademark information from the document:

BASIC DETAILS:
- Trademark name/title
- Registration number or application number
- Registration date or application date

COMPANY/APPLICANT INFORMATION:
- Company/applicant name
- Full address including city, state, country
- Business type or firm type

TRADEMARK DETAILS:
- "text_in_logo": Extract ONLY the company name as it appears within the logo/trademark itself. This should be the stylized text that forms the visual brand identity. Look for text that is designed as part of the logo graphics, often in special fonts, colors, or styling. If no text is visible in the logo design, return empty string.
- "logo_description": Provide a brief 5-8 word visual description of what the logo looks like (e.g., "red circle with white text", "geometric blue triangular design", "stylized bird with wings spread", "minimalist black and white lines")
- Business category or classification
- Description of goods/services

CONTACT/LEGAL INFORMATION:
- Contact person or legal representative
- Phone, email, website if available
- Legal status or registration status

IMPORTANT DISTINCTION:
- "name": Company/applicant legal name (from application forms)
- "text_in_logo": Only the text that appears AS PART OF the logo design itself (stylized company name within the trademark symbol)

Return the JSON in this exact structure:

{{
  "name": "string",
  "address": "string", 
  "city": "string",
  "firm_type": "string",
  "date": "string",
  "text_in_logo": "string",
  "logo_description": "string",
  "registration_number": "string",
  "business_category": "string",
  "contact_person": "string",
  "phone": "string",
  "email": "string",
  "website": "string",
  "legal_status": "string",
  "description": "string",
  "state": "string",
  "country": "string"
}}

IMPORTANT: Return ONLY the JSON object. Ensure all syntax is valid JSON.
"""),
            ("human", "Extract trademark data from this text and return structured JSON:\n{text}")
        ])

    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> list:
        """
        Extract trademark data from PDF bytes by processing each page as an image
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            all_trademarks = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to base64
                base64_image = base64.b64encode(img_data).decode("utf-8")
                data_url = f"data:image/png;base64,{base64_image}"
                
                # Extract text from image
                extracted_text = self.extract_from_base64_image(base64_image)
                
                if extracted_text and extracted_text.get("text"):
                    # Extract structured trademark data
                    trademark_data = self.extract_trademark_data_from_text(extracted_text["text"])
                    
                    if trademark_data:
                        # Add page number to the data
                        trademark_data["page_number"] = page_num + 1
                        all_trademarks.append(trademark_data)
            
            pdf_document.close()
            return all_trademarks
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    def extract_from_base64_image(self, base64_image: str) -> dict:
        """
        Extract text from a base64 encoded image using the vision model
        """
        try:
            # Remove data URL prefix if present
            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',')[1]
            
            # Create data URL for the image
            data_url = f"data:image/png;base64,{base64_image}"
            
            # Use the text extraction model
            chain = self.text_extract_prompt_template | self.model | StrOutputParser()
            
            result = chain.invoke({"image_url": data_url})
            
            return {
                "text": result,
                "status": "success"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text from image: {str(e)}")

    def extract_trademark_data_from_text(self, text: str) -> dict:
        """
        Extract structured trademark data from text using the structured extraction model
        """
        try:
            if not text or not text.strip():
                return {}
            
            # Use the structured extraction model
            chain = self.prompt_template | self.model2 | StrOutputParser()
            
            result = chain.invoke({"text": text})
            
            # Try to parse as JSON
            try:
                # Clean the result to extract JSON
                json_text = result.strip()
                
                # Remove any markdown code blocks if present
                if json_text.startswith('```json'):
                    json_text = json_text[7:]
                if json_text.startswith('```'):
                    json_text = json_text[3:]
                if json_text.endswith('```'):
                    json_text = json_text[:-3]
                
                json_text = json_text.strip()
                
                # Parse JSON
                trademark_data = json.loads(json_text)
                
                # Ensure all required fields exist with default values
                required_fields = {
                    "name": "",
                    "address": "",
                    "city": "",
                    "firm_type": "",
                    "date": "",
                    "text_in_logo": "",
                    "logo_description": "",
                    "registration_number": "",
                    "business_category": "",
                    "contact_person": "",
                    "phone": "",
                    "email": "",
                    "website": "",
                    "legal_status": "",
                    "description": "",
                    "state": "",
                    "country": ""
                }
                
                # Fill in missing fields with defaults
                for field, default_value in required_fields.items():
                    if field not in trademark_data:
                        trademark_data[field] = default_value
                    elif trademark_data[field] is None:
                        trademark_data[field] = default_value
                
                return trademark_data
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return a basic structure with the raw text
                return {
                    "name": "",
                    "address": "",
                    "city": "",
                    "firm_type": "",
                    "date": "",
                    "text_in_logo": "",
                    "logo_description": "",
                    "registration_number": "",
                    "business_category": "",
                    "contact_person": "",
                    "phone": "",
                    "email": "",
                    "website": "",
                    "legal_status": "",
                    "description": text[:500] if text else "",  # Use first 500 chars as description
                    "state": "",
                    "country": "",
                    "raw_extraction": result  # Keep raw result for debugging
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting trademark data: {str(e)}")

    def extract_from_text(self, text: str) -> dict:
        """
        Extract trademark data from plain text
        """
        return self.extract_trademark_data_from_text(text)

    def extract_from_image_file(self, image_path: str) -> dict:
        """
        Extract trademark data from an image file
        """
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                # Extract text from image
                extracted_text = self.extract_from_base64_image(base64_image)
                
                if extracted_text and extracted_text.get("text"):
                    # Extract structured trademark data
                    return self.extract_trademark_data_from_text(extracted_text["text"])
                
                return {}
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image file: {str(e)}")

 
