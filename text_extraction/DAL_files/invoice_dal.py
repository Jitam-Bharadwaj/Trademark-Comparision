import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from schemas.invoice_schemas import InvoiceData
import fitz
import base64
import json
import re
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends


class SimpleInvoiceExtractor:
    def __init__(self, groq_api_key: str):
        self.model = ChatGroq(
            temperature=0.1,
            groq_api_key=groq_api_key,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        self.model2 = ChatGroq(
            temperature=0.1,
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
        )

        self.text_extract_prompt_template = self.image_text_extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract all text content from this document exactly as it appears.\n      Maintain the original layout and formatting as much as possible.\n      Pay special attention to:\n      1. Table structures and numerical data\n      2. Invoice-specific fields like dates, amounts, and IDs\n      3. Vendor and customer information\n      4. Line items with quantities and prices"),

            ("human",
                [
                    {"type":"text", "text": "Extract all text from this document with high accuracy:"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}}
                ])
        ])

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert invoice data extraction assistant. You must return ONLY a valid JSON object with no additional text, formatting, or explanations.

CRITICAL RULES:
1. Return ONLY the JSON object - no markdown, no code blocks, no explanations
2. Ensure all JSON syntax is correct - no trailing commas, proper quotes, valid structure
3. Use consistent field names throughout
4. All string values must be properly quoted
5. All numeric values should be strings for consistency
6. Do not include any comments or extra text

DYNAMIC FIELD NAMES:
- For all fields in lineItems (and anywhere else in the JSON where the document provides a field name), use the exact field names as they appear in the document.
- For example, if the document uses "Unit price", "MRP", "Rate", "Qty", "Quantity", "Description", etc., use those exact keys in the output JSON.
- Do not standardize, translate, or change the field names.
- If a field is missing in the document, keep the value as null.

Extract the following information from the document:

BASIC DETAILS:
- Invoice number, date, due date
- Reference numbers or codes

VENDOR INFORMATION:
- Name, address, tax ID, contact information

CUSTOMER INFORMATION:
- Name, address, contact information

LINE ITEMS:
- Description, quantity, unit price/rate, amount
- Use original field names from document (e.g., "qty" vs "quantity", "rate" vs "unitPrice")

FINANCIAL DETAILS:
- Subtotal, taxes, discounts, shipping, total
- Currency and tax information

PAYMENT INFORMATION:
- Payment method, terms, bank details
- Payment links or early discount terms

METADATA:
- Language detection (ISO 639-1)
- Country identification
- Confidence scores for extraction accuracy
- Financial validation and audit results

Return the JSON in this exact structure:

{{
  "invoiceNumber": "string",
  "date": "string",
  "dueDate": "string",
  "vendor": {{
    "name": "string",
    "address": "string",
    "taxId": "string",
    "contactInfo": "string"
  }},
  "customer": {{
    "name": "string",
    "address": "string",
    "contactInfo": "string"
  }},
  "lineItems": [
    {{
      "description": "string",
      "qty": "string",
      "rate": "string",
      "amount": "string"
      // The keys here should match exactly as they appear in the document, e.g., "Unit price", "MRP", "Rate", etc.
    }}
  ],
  "financials": {{
    "subtotal": "string",
    "discount": "string",
    "tax": "string",
    "taxRate": "string",
    "shipping": "string",
    "total": "string",
    "currency": "string",
    "taxName": "string"
  }},
  "payment": {{
    "method": "string",
    "terms": "string",
    "bankDetails": {{
      "accountNumber": "string",
      "routingNumber": "string",
      "iban": "string",
      "swift": "string",
      "bankName": "string"
    }},
    "paymentLink": "string"
  }},
  "meta": {{
    "language": "string",
    "languageName": "string",
    "country": "string",
    "countryCode": "string",
    "confidence": {{
      "overall": 0,
      "fields": {{}}
    }},
    "audit": {{
      "status": "string",
      "issues": [],
      "taxCompliance": {{
        "status": "string",
        "details": "string"
      }}
    }},
    "suggestions": {{
      "invoiceType": "string",
      "categories": [
        {{"name": "string", "confidence": 0}}
      ],
      "vendorTypes": [],
      "selectedCategory": "string",
      "selectedVendorType": null
    }}
  }}
}}

IMPORTANT: Return ONLY the JSON object. Ensure all syntax is valid JSON.
"""),
            ("human", "Extract invoice data from this text and return structured JSON:\n{text}")
        ])

    def clean_json_response(self, response: str) -> str:
        """Clean the response to extract pure JSON"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = response.strip()
        
        # Try to find JSON object boundaries
        start_idx = response.find('{')
        if start_idx != -1:
            # Find the last closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                response = response[start_idx:end_idx + 1]
        
        return response

    def classify_document(self, text: str) -> str:
        lower_text = text.lower()
        invoice_patterns = [
            "invoice", "bill to", "factura", "rechnung", "facture", "fattura", "发票", "インボイス",
            "فاتورة", "חשבונית", "счет", "tax invoice", "billing statement", "payment due",
            "invoice number", "invoice no", "invoice #", "inv #", "inv no", "invoice date"
        ]
        receipt_patterns = [
            "receipt", "payment received", "paid", "payment confirmation", "proof of payment",
            "recibo", "quittung", "reçu", "ricevuta", "收据", "領収書", "إيصال", "קבלה", "квитанция",
            "thank you for your purchase", "cash receipt", "payment receipt"
        ]
        po_patterns = [
            "purchase order", "p.o.", "p/o", "order confirmation", "order form",
            "orden de compra", "bestellung", "bon de commande", "ordine d'acquisto", "采购订单",
            "注文書", "أمر شراء", "הזמנת רכש", "заказ на покупку"
        ]
        quote_patterns = [
            "quote", "estimate", "quotation", "proposal", "pro forma", "proforma",
            "presupuesto", "angebot", "devis", "preventivo", "报价", "見積もり",
            "عرض أسعار", "הצעת מחיר", "коммерческое предложение"
        ]
        statement_patterns = [
            "statement", "account statement", "statement of account", "monthly statement",
            "estado de cuenta", "kontoauszug", "relevé de compte", "estratto conto", "对账单",
            "取引明細書", "كشف حساب", "דף חשבון", "выписка по счету"
        ]
        creditNote_patterns = [
            "credit note", "credit memo", "credit memorandum", "refund",
            "nota de crédito", "gutschrift", "note de crédit", "nota di credito", "贷记通知单",
            "クレジットノート", "إشعار دائن", "הודעת זיכוי", "кредитное авизо"
        ]
        
        for pattern in invoice_patterns:
            if pattern in lower_text:
                return "invoice"
        for pattern in receipt_patterns:
            if pattern in lower_text:
                return "receipt"
        for pattern in po_patterns:
            if pattern in lower_text:
                return "purchase_order"
        for pattern in quote_patterns:
            if pattern in lower_text:
                return "quote"
        for pattern in statement_patterns:
            if pattern in lower_text:
                return "statement"
        for pattern in creditNote_patterns:
            if pattern in lower_text:
                return "credit_note"
        
        if (
            ("total" in lower_text and ("due" in lower_text or "amount" in lower_text)) or
            ("payment" in lower_text and "terms" in lower_text) or
            ("tax" in lower_text and "subtotal" in lower_text)
        ):
            return "invoice"
        
        import re
        if re.search(r"inv[^a-z]", lower_text) and re.search(r"\d{4,}", lower_text):
            return "invoice"
        return "invoice"    

    def extract_invoice_fromate_from_text(self, text: str, doctype: str):
        try:
            # Get raw response from model
            chain = self.prompt_template | self.model2
            response = chain.invoke(
                {"text": text, "documentType": doctype},
                config={"return_token_usage": True}
            )
            
            # Clean the response to extract pure JSON
            clean_response = self.clean_json_response(response.content)

            # Parse JSON manually with better error handling
            try:
                parsed_data = json.loads(clean_response)
                return parsed_data
            except json.JSONDecodeError as json_error:
                raise HTTPException(status_code=500, detail=str(json_error))
                
        except Exception as e:
            print(f"Error processing text: {e}")
            raise HTTPException(status_code=500, detail="Internal processing error")


    def fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unescaped quotes in strings (basic fix)
        # This is a simple approach - for more complex cases, you might need a more sophisticated parser
        
        return json_str
    
    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> InvoiceData:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("No pages found in PDF")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        return self.extract_from_base64_image(base64_image)

    def extract_from_base64_image(self, base64_image: str) -> InvoiceData:
        try:
            chain = (
                self.text_extract_prompt_template 
                | self.model 
                | {"text": StrOutputParser(), "metadata": lambda x: x}
            )
            result = chain.invoke(
                {"image_url": f"data:image/png;base64,{base64_image}"},
                config={"return_token_usage": True}
            )
            
            token_usage = result["metadata"].usage_metadata
            
            return {"text": result["text"]}
            
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            raise