import os
import json
import math
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any
import zipfile
import tempfile

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cloud deployment
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

# Try to import OpenCV with fallback for headless environments
try:
    import cv2
    CV2_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ OpenCV not available: {e}")
    CV2_AVAILABLE = False
    # Create a mock cv2 module for basic functionality
    class MockCV2:
        @staticmethod
        def imread(path):
            return None
        @staticmethod
        def cvtColor(img, code):
            return img
        @staticmethod
        def findContours(img, mode, method):
            return [], None
        @staticmethod
        def contourArea(contour):
            return 0
        @staticmethod
        def boundingRect(contour):
            return (0, 0, 100, 100)
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 0
        COLOR_BGR2GRAY = 6
    cv2 = MockCV2()

# Try to import other dependencies with fallbacks
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False
    def cosine_similarity(a, b):
        return np.array([[0.0]])

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import timm
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    T = None
    timm = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ FAISS not available: {e}")
    FAISS_AVAILABLE = False
    faiss = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TQDM not available: {e}")
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None):
        return iterable

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ EasyOCR not available: {e}")
    OCR_AVAILABLE = False
    easyocr = None

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sentence Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# PDF processing
try:
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PDF processing not available: {e}")
    PDF_AVAILABLE = False
    convert_from_path = None
    fitz = None

class InteractiveTrademarkSearch:
    def __init__(self):
        # Check if we have the minimum required dependencies
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - visual comparison disabled")
            self.device = "cpu"
            self.vit_model = None
            self.efficient_model = None
            self.text_encoder = None
            self.ocr_reader = None
            self.transform = None
            self.index = None
            self.paths = []
            self.metadata = []
            self.features = []
            return
        
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        print("🔧 Initializing AI models...")
        
        # Initialize OCR
        if OCR_AVAILABLE and easyocr:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() if torch else False, verbose=False)
                print("✅ OCR reader loaded")
            except Exception as e:
                print(f"⚠️ OCR loading failed: {e}")
                self.ocr_reader = None
        else:
            print("⚠️ OCR not available")
            self.ocr_reader = None
        
        # Initialize vision models
        if TORCH_AVAILABLE and timm and nn:
            try:
                self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
                self.vit_model.head = nn.Identity()
                self.vit_model.eval().to(self.device)
                print("✅ Vision Transformer loaded")
            except Exception as e:
                print(f"⚠️ ViT loading failed: {e}")
                self.vit_model = None
            
            try:
                self.efficient_model = timm.create_model('efficientnet_b0', pretrained=True)
                self.efficient_model.classifier = nn.Identity()
                self.efficient_model.eval().to(self.device)
                print("✅ EfficientNet loaded")
            except Exception as e:
                print(f"⚠️ EfficientNet loading failed: {e}")
                self.efficient_model = None
        else:
            print("⚠️ Vision models not available")
            self.vit_model = None
            self.efficient_model = None
        
        # Text encoder
        if TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                print("✅ Text encoder loaded")
            except Exception as e:
                print(f"⚠️ Text encoder loading failed: {e}")
                self.text_encoder = None
        else:
            print("⚠️ Text encoder not available")
            self.text_encoder = None
        
        # Image transforms
        if TORCH_AVAILABLE and T:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = None
        
        # Storage
        self.index = None
        self.paths = []
        self.metadata = []
        self.features = []
        
        print("🎉 System initialized successfully!")
    
    def ensure_dir(self, p: str):
        Path(p).mkdir(parents=True, exist_ok=True)
    
    def pdf_to_images(self, pdf_path: str, out_dir="extracted_images") -> List[str]:
        """Convert PDF to high-quality images"""
        if not PDF_AVAILABLE:
            print("❌ PDF processing not available")
            return []
            
        self.ensure_dir(out_dir)
        base = Path(pdf_path).stem
        
        try:
            # Use pdf2image for best quality
            if convert_from_path:
                pages = convert_from_path(pdf_path, dpi=300, fmt='PNG')
                paths = []
                for i, page in enumerate(pages, start=1):
                    p = f"{out_dir}/{base}_page_{i:03d}.png"
                    page = page.convert('RGB')
                    # Enhance for logo detection
                    enhancer = ImageEnhance.Sharpness(page)
                    page = enhancer.enhance(1.1)
                    page.save(p, "PNG", optimize=True)
                    paths.append(p)
                print(f"📄 Extracted {len(paths)} pages from {base}")
                return paths
            else:
                raise Exception("pdf2image not available")
            
        except Exception as e:
            print(f"⚠️ pdf2image failed: {e}, trying PyMuPDF...")
            try:
                # Fallback to PyMuPDF
                if fitz:
                    doc = fitz.open(pdf_path)
                    paths = []
                    zoom = 300 / 72.0
                    mtx = fitz.Matrix(zoom, zoom)
                    
                    for i, page in enumerate(doc, start=1):
                        pm = page.get_pixmap(matrix=mtx, alpha=False)
                        p = f"{out_dir}/{base}_page_{i:03d}.png"
                        pm.save(p)
                        paths.append(p)
                    doc.close()
                    print(f"📄 Extracted {len(paths)} pages from {base} (fallback)")
                    return paths
                else:
                    raise Exception("PyMuPDF not available")
            except Exception as e2:
                print(f"❌ Both PDF methods failed: {e2}")
                return []
    
    def preprocess_logo(self, image_path: str) -> Tuple[Image.Image, Dict]:
        """Extract logo region and metadata"""
        try:
            # Load image
            if CV2_AVAILABLE:
                img_cv = cv2.imread(image_path)
                if img_cv is None:
                    raise ValueError(f"Could not load image: {image_path}")
                    
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Find main content region
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                logo_bbox = None
                
                if contours:
                    # Find the most significant contour
                    areas = [cv2.contourArea(c) for c in contours]
                    if areas:
                        largest_idx = np.argmax(areas)
                        largest_contour = contours[largest_idx]
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Check if it's a reasonable logo region
                        img_area = img_cv.shape[0] * img_cv.shape[1]
                        contour_area = w * h
                        if 0.02 < contour_area / img_area < 0.9:
                            logo_bbox = (x, y, w, h)
            else:
                # Fallback when OpenCV is not available
                logo_bbox = None
            
            # Extract text using OCR
            extracted_text = ""
            try:
                if self.ocr_reader:
                    text_results = self.ocr_reader.readtext(image_path, paragraph=False)
                    texts = [result[1] for result in text_results if result[2] > 0.3]  # Confidence > 30%
                    extracted_text = " ".join(texts)
            except Exception as ocr_e:
                print(f"OCR failed for {image_path}: {ocr_e}")
            
            # Load PIL image
            pil_img = Image.open(image_path).convert('RGB')
            
            # Crop to logo region if found
            if logo_bbox:
                x, y, w, h = logo_bbox
                # Add small padding
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(pil_img.width - x, w + 2*pad)
                h = min(pil_img.height - y, h + 2*pad)
                pil_img = pil_img.crop((x, y, x+w, y+h))
            
            metadata = {
                'text': extracted_text.strip(),
                'bbox': logo_bbox,
                'has_text': len(extracted_text.strip()) > 0,
                'original_path': image_path
            }
            
            return pil_img, metadata
            
        except Exception as e:
            print(f"⚠️ Preprocessing failed for {image_path}: {e}")
            # Return original image with minimal metadata
            pil_img = Image.open(image_path).convert('RGB')
            return pil_img, {'text': '', 'bbox': None, 'has_text': False, 'original_path': image_path}
    
    def extract_features(self, img: Image.Image) -> np.ndarray:
        """Extract visual features using available models"""
        if not TORCH_AVAILABLE or not self.transform:
            # Fallback: simple image statistics
            img_array = np.array(img.resize((64, 64)))
            combined = img_array.flatten()
            print("⚠️ Using fallback features (basic image statistics)")
            return combined.astype('float32')
        
        # Use torch.no_grad() decorator if available
        if torch:
            @torch.no_grad()
            def _extract_with_torch():
                x = self.transform(img).unsqueeze(0).to(self.device)
                features = []
                
                # ViT features
                if self.vit_model is not None:
                    try:
                        vit_feat = self.vit_model(x).squeeze(0).detach().cpu().numpy()
                        features.append(vit_feat)
                    except Exception as e:
                        print(f"ViT extraction failed: {e}")
                
                # EfficientNet features
                if self.efficient_model is not None:
                    try:
                        eff_feat = self.efficient_model(x).squeeze(0).detach().cpu().numpy()
                        features.append(eff_feat)
                    except Exception as e:
                        print(f"EfficientNet extraction failed: {e}")
                
                # Combine features
                if features:
                    combined = np.concatenate(features)
                else:
                    # Fallback: simple image statistics
                    img_array = np.array(img.resize((64, 64)))
                    combined = img_array.flatten()
                    print("⚠️ Using fallback features (basic image statistics)")
                
                return combined.astype('float32')
            
            return _extract_with_torch()
        else:
            # Fallback: simple image statistics
            img_array = np.array(img.resize((64, 64)))
            combined = img_array.flatten()
            print("⚠️ Using fallback features (basic image statistics)")
            return combined.astype('float32')
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features"""
        if not text.strip() or self.text_encoder is None:
            return np.zeros(384, dtype='float32')
        
        try:
            embedding = self.text_encoder.encode(text, convert_to_numpy=True)
            return embedding.astype('float32')
        except Exception as e:
            print(f"Text encoding failed: {e}")
            return np.zeros(384, dtype='float32')
    
    def process_uploaded_pdfs(self, uploaded_files: List[bytes], filenames: List[str]) -> List[str]:
        """Process uploaded PDF files"""
        pdf_files = [(f, c) for f, c in zip(filenames, uploaded_files) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files found in upload!")
            return []
        
        print(f"📁 Processing {len(pdf_files)} PDF file(s)...")
        
        # Save uploaded files to temporary directory
        temp_dir = tempfile.mkdtemp()
        all_images = []
        
        for filename, content in pdf_files:
            temp_pdf_path = os.path.join(temp_dir, filename)
            with open(temp_pdf_path, 'wb') as f:
                f.write(content)
            
            try:
                images = self.pdf_to_images(temp_pdf_path, out_dir=os.path.join(temp_dir, "extracted_images"))
                all_images.extend(images)
                print(f"✅ {filename}: {len(images)} pages extracted")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
        
        return all_images
    
    def build_index(self, image_paths: List[str]):
        """Build searchable index from images"""
        if not FAISS_AVAILABLE or not faiss:
            print("❌ FAISS not available - cannot build search index")
            return
            
        print("🔍 Building search index...")
        
        visual_features = []
        text_features = []
        metadata_list = []
        valid_paths = []
        
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                # Preprocess image
                logo_img, metadata = self.preprocess_logo(path)
                
                # Extract visual features
                visual_feat = self.extract_features(logo_img)
                
                # Extract text features
                text_feat = self.extract_text_features(metadata['text'])
                
                # Store everything
                visual_features.append(visual_feat)
                text_features.append(text_feat)
                metadata_list.append(metadata)
                valid_paths.append(path)
                
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")
                continue
        
        if not visual_features:
            print("❌ No valid features extracted from any image!")
            return
        
        # Combine visual and text features
        combined_features = []
        for vis_feat, txt_feat in zip(visual_features, text_features):
            # Combine visual and text features with weights
            combined = np.concatenate([vis_feat * 0.8, txt_feat * 0.2])  # Favor visual
            combined_features.append(combined)
        
        # Build FAISS index
        feature_matrix = np.vstack(combined_features).astype('float32')
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        feature_matrix = feature_matrix / (norms + 1e-8)
        
        # Create index
        self.index = faiss.IndexFlatIP(feature_matrix.shape[1])  # Inner product for normalized vectors
        self.index.add(feature_matrix)
        
        # Store metadata
        self.paths = valid_paths
        self.metadata = metadata_list
        self.features = combined_features
        
        print(f"🎉 Index built successfully with {len(valid_paths)} images!")
        
        # Show sample extracted text
        text_samples = [m['text'] for m in metadata_list if m['text']][:5]
        if text_samples:
            print(f"📝 Sample extracted text: {text_samples}")
    
    def search_similar(self, query_image_path: str, k: int = 5) -> Dict:
        """Find similar trademarks"""
        if not FAISS_AVAILABLE or not faiss:
            return {'matches': [], 'scores': [], 'error': 'FAISS not available'}
            
        if self.index is None:
            return {'matches': [], 'scores': [], 'error': 'Index not built! Please process PDFs first.'}
        
        try:
            # Process query image
            query_img, query_metadata = self.preprocess_logo(query_image_path)
            
            # Extract features
            visual_feat = self.extract_features(query_img)
            text_feat = self.extract_text_features(query_metadata['text'])
            
            # Combine features (same weighting as index)
            query_combined = np.concatenate([visual_feat * 0.8, text_feat * 0.2])
            
            # Normalize
            query_combined = query_combined / (np.linalg.norm(query_combined) + 1e-8)
            
            # Search
            scores, indices = self.index.search(query_combined.reshape(1, -1), k)
            
            results = {
                'query_metadata': query_metadata,
                'matches': [self.paths[i] for i in indices[0]],
                'scores': scores[0].tolist(),
                'match_metadata': [self.metadata[i] for i in indices[0]]
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {'matches': [], 'scores': [], 'error': str(e)}
