#!/usr/bin/env python3
"""
Main Trademark Analysis System
Combines both text-based extraction and visual logo comparison capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import json
from datetime import datetime
import base64
from io import BytesIO
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import both systems
from text_extraction.DAL_files.trademark_dal import TrademarkExtractor
from text_extraction.DAL_files.trademark_comparison import TrademarkComparator
from text_extraction.DAL_files.csv_manager import TrademarkCSVManager
from visual_comparison.DAL_files.trademark_search import InteractiveTrademarkSearch

# Configure Streamlit page
st.set_page_config(
    page_title="🏷️ Trademark Analysis System",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .system-card {
        background-color: #000000;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .system-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .similarity-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .similarity-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
    }
    .similarity-low {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extracted_trademarks' not in st.session_state:
    st.session_state.extracted_trademarks = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'csv_loaded' not in st.session_state:
    st.session_state.csv_loaded = False
if 'current_system' not in st.session_state:
    st.session_state.current_system = None
if 'trademark_system' not in st.session_state:
    st.session_state.trademark_system = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False

# Initialize classes
@st.cache_resource
def initialize_text_components():
    """Initialize text extraction components"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "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")
        extractor = TrademarkExtractor(groq_api_key=groq_api_key)
        comparator = TrademarkComparator()
        csv_manager = TrademarkCSVManager()
        return extractor, comparator, csv_manager
    except Exception as e:
        st.error(f"Error initializing text components: {str(e)}")
        return None, None, None

@st.cache_resource
def initialize_visual_components():
    """Initialize visual comparison components"""
    try:
        return InteractiveTrademarkSearch()
    except Exception as e:
        st.error(f"Error initializing visual components: {str(e)}")
        return None

def show_text_extraction_ui():
    """Text extraction and comparison UI"""
    extractor, comparator, csv_manager = initialize_text_components()
    
    if not extractor:
        st.error("❌ Failed to initialize text extraction components")
        return
    
    # Main title
    st.markdown('<h1 class="main-header">📄 Text-Based Extraction & Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Configuration")
        
        # CSV Database Section
        st.subheader("🗃️ Trademark Database")
        
        # Upload CSV database
        uploaded_csv = st.file_uploader(
            "Upload Existing Trademark Database (CSV)",
            type=['csv'],
            help="Upload a CSV file containing existing trademark data for comparison"
        )
        
        if uploaded_csv is not None:
            try:
                # Save uploaded CSV
                csv_content = uploaded_csv.read()
                with open(csv_manager.csv_file_path, 'wb') as f:
                    f.write(csv_content)
                
                # Load into comparator
                comparator.load_existing_trademarks_from_csv(csv_manager.csv_file_path)
                st.session_state.csv_loaded = True
                st.success(f"✅ Loaded {len(comparator.existing_trademarks)} trademarks from CSV")
                
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")
        
        # Check if default CSV exists
        elif os.path.exists(csv_manager.csv_file_path):
            try:
                comparator.load_existing_trademarks_from_csv(csv_manager.csv_file_path)
                st.session_state.csv_loaded = True
                st.info(f"📊 Using existing database: {len(comparator.existing_trademarks)} trademarks")
            except Exception as e:
                st.warning("Default CSV file found but could not be loaded")
        
        # Similarity threshold
        st.subheader("⚙️ Comparison Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=30.0,
            max_value=95.0,
            value=50.0,
            step=5.0,
            help="Minimum similarity percentage to consider as a potential match"
        )
    
    # Main content area
    if not st.session_state.csv_loaded:
        st.warning("⚠️ Please upload a CSV database in the sidebar to enable comparison functionality.")
    
    # PDF Upload Section
    st.header("📄 PDF Trademark Extraction")
    
    uploaded_pdf = st.file_uploader(
        "Upload PDF containing trademark information",
        type=['pdf'],
        help="Upload a PDF file to extract trademark information and compare with existing database"
    )
    
    if uploaded_pdf is not None and extractor is not None:
        with st.spinner("🔍 Extracting trademark data from PDF..."):
            try:
                # Read PDF content
                pdf_bytes = uploaded_pdf.read()
                
                # Extract trademark data
                trademarks_data = extractor.extract_from_pdf_bytes(pdf_bytes)
                st.session_state.extracted_trademarks = trademarks_data
                
                # Show extraction results
                st.success(f"✅ Successfully extracted {len(trademarks_data)} trademark(s) from {uploaded_pdf.name}")
                
                # Display extracted trademarks
                st.header("🎯 Extracted Trademark Information")
                
                for i, trademark in enumerate(trademarks_data):
                    with st.expander(f"📋 Trademark {i+1} (Page {trademark.get('page_number', i+1)})", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Company Information**")
                            st.write(f"**Name:** {trademark.get('name', 'N/A')}")
                            st.write(f"**Address:** {trademark.get('address', 'N/A')}")
                            st.write(f"**City:** {trademark.get('city', 'N/A')}")
                            st.write(f"**State:** {trademark.get('state', 'N/A')}")
                            st.write(f"**Country:** {trademark.get('country', 'N/A')}")
                        
                        with col2:
                            st.markdown("**Trademark Details**")
                            st.write(f"**Text in Logo:** {trademark.get('text_in_logo', 'N/A')}")
                            st.write(f"**Logo Description:** {trademark.get('logo_description', 'N/A')}")
                            st.write(f"**Registration Number:** {trademark.get('registration_number', 'N/A')}")
                            st.write(f"**Business Category:** {trademark.get('business_category', 'N/A')}")
                            st.write(f"**Date:** {trademark.get('date', 'N/A')}")
                        
                        with col3:
                            st.markdown("**Contact & Legal**")
                            st.write(f"**Contact Person:** {trademark.get('contact_person', 'N/A')}")
                            st.write(f"**Phone:** {trademark.get('phone', 'N/A')}")
                            st.write(f"**Email:** {trademark.get('email', 'N/A')}")
                            st.write(f"**Website:** {trademark.get('website', 'N/A')}")
                            st.write(f"**Legal Status:** {trademark.get('legal_status', 'N/A')}")
                            st.write(f"**Firm Type:** {trademark.get('firm_type', 'N/A')}")
                        
                        if trademark.get('description'):
                            st.markdown("**Description**")
                            st.write(trademark.get('description'))
                        
                        # JSON view
                        with st.expander("🔍 View Raw JSON Data"):
                            st.json(trademark)
                
                # Comparison Section
                if st.session_state.csv_loaded and comparator is not None:
                    st.header("🔍 Similarity Analysis")
                    
                    comparison_results = []
                    for i, trademark in enumerate(trademarks_data):
                        with st.spinner(f"Comparing trademark {i+1} with database..."):
                            comparison_report = comparator.generate_comparison_report(
                                trademark, 
                                similarity_threshold
                            )
                            comparison_results.append(comparison_report)
                    
                    st.session_state.comparison_results = comparison_results
                    
                    # Display comparison results
                    for i, result in enumerate(comparison_results):
                        st.subheader(f"🎯 Trademark {i+1} Comparison Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Similar Trademarks Found", result['similar_trademarks_found'])
                        with col2:
                            st.metric("Total Database Records", result['total_existing_trademarks'])
                        with col3:
                            st.metric("Similarity Threshold", f"{result['similarity_threshold']:.1f}%")
                        with col4:
                            if result['similar_trademarks_found'] > 0:
                                max_similarity = max([match['similarity_score'] for match in result['matches']])
                                st.metric("Highest Similarity", f"{max_similarity:.1f}%")
                        
                        # Show matches if any
                        if result['similar_trademarks_found'] > 0:
                            st.markdown("### 🚨 Potential Conflicts Found")
                            
                            for j, match in enumerate(result['matches']):
                                similarity_score = match['similarity_score']
                                similarity_level = match['similarity_level']
                                
                                # Color code based on similarity level
                                if similarity_score >= 85:
                                    css_class = "similarity-high"
                                    icon = "🔴"
                                elif similarity_score >= 70:
                                    css_class = "similarity-medium"
                                    icon = "🟡"
                                else:
                                    css_class = "similarity-low"
                                    icon = "🟣"
                                
                                with st.container():
                                    st.markdown(f"""
                                    <div class="{css_class}">
                                        <h4>{icon} Match {j+1}: {similarity_level}</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Match details
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown("**📄 From PDF (New)**")
                                        new_trademark = result['new_trademark']
                                        st.write(f"**Company Name:** {new_trademark.get('name', 'N/A')}")
                                        st.write(f"**Trademark Text:** {new_trademark.get('trademark', 'N/A')}")
                                        st.write(f"**Application No.:** {new_trademark.get('application_no', 'N/A')}")
                                        st.write(f"**Class:** {new_trademark.get('class', 'N/A')}")
                                        st.write(f"**Status:** {new_trademark.get('status', 'N/A')}")
                                    
                                    with col2:
                                        st.markdown("**🗃️ From CSV (Existing)**")
                                        existing = match['existing_trademark']
                                        st.write(f"**Company Name:** {existing.get('Client / Applicant', 'N/A')}")
                                        st.write(f"**Trademark Text:** {existing.get('Trademark', 'N/A')}")
                                        st.write(f"**Application No.:** {existing.get('Application No.', 'N/A')}")
                                        st.write(f"**Class:** {existing.get('Class', 'N/A')}")
                                        st.write(f"**Status:** {existing.get('Status', 'N/A')}")
                                    
                                    with col3:
                                        st.markdown("**📊 Similarity Breakdown**")
                                        st.write(f"**Overall Score:** {similarity_score:.1f}%")
                                        st.write(f"**Similarity Type:** {match['similarity_type'].title()}")
                                        
                                        # Detailed scores
                                        detailed = match['detailed_scores']
                                        name_score = detailed['name_comparison']['overall']
                                        logo_score = detailed['logo_comparison']['overall']
                                        
                                        st.write(f"**Name Similarity:** {name_score:.1f}%")
                                        st.write(f"**Logo Similarity:** {logo_score:.1f}%")
                        else:
                            st.success("✅ No similar trademarks found in the database above the similarity threshold.")
                        
                        st.markdown("---")
                
                else:
                    st.info("💡 Upload a CSV database to enable similarity comparison.")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

def initialize_visual_system():
    """Initialize the trademark search system"""
    if st.session_state.trademark_system is None:
        with st.spinner("🔧 Initializing AI models... This may take a few minutes on first run."):
            st.session_state.trademark_system = InteractiveTrademarkSearch()
        return True
    return True

def show_search_results(results, max_results=5, query_image_bytes=None):
    """Display search results with enhanced visualization"""
    if 'error' in results:
        st.error(f"❌ Search error: {results['error']}")
        return
    
    matches = results.get('matches', [])
    scores = results.get('scores', [])
    query_metadata = results.get('query_metadata', {})
    
    if not matches:
        st.warning("❌ No matches found!")
        return
    
    st.success(f"🔍 Found {len(matches)} similar trademarks!")
    
    # Show query info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Query Image")
        try:
            if query_image_bytes:
                # Display the query image from bytes to avoid file path issues
                query_img = Image.open(query_image_bytes)
                st.image(query_img, width='stretch')
            else:
                st.info("Query image not available for display")
        except Exception as e:
            st.error(f"Error loading query image: {e}")
    
    with col2:
        st.subheader("📋 Query Details")
        query_text = query_metadata.get('text', '')
        if query_text:
            st.write(f"**Detected Text:** {query_text}")
        else:
            st.write("**Detected Text:** None")
        
        bbox = query_metadata.get('bbox', None)
        if bbox:
            st.write(f"**Logo Region:** {bbox}")
        else:
            st.write("**Logo Region:** Not detected")
    
    # Show matches in a more robust way
    st.subheader(f"🔍 Top {min(len(matches), max_results)} Similar Trademarks")
    
    # Display results in a more stable format
    for i, (match_path, score, match_meta) in enumerate(zip(matches[:max_results], scores[:max_results], results.get('match_metadata', [{}]*len(matches)))):
        with st.expander(f"Match {i+1} - Similarity: {score*100:.1f}%", expanded=(i < 3)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                try:
                    if os.path.exists(match_path):
                        match_img = Image.open(match_path)
                        st.image(match_img, width='stretch')
                    else:
                        st.error(f"Image not found: {Path(match_path).name}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            with col2:
                # Similarity score with color coding
                similarity_percent = score * 100
                if similarity_percent > 70:
                    color = "🟢"
                    status = "High Similarity"
                elif similarity_percent > 40:
                    color = "🟡"
                    status = "Medium Similarity"
                else:
                    color = "🔴"
                    status = "Low Similarity"
                
                st.write(f"**{color} {status}**")
                st.write(f"**Similarity:** {similarity_percent:.1f}%")
                
                # Show extracted text if available
                match_text = match_meta.get('text', '')
                if match_text:
                    st.write(f"**Text:** {match_text[:100]}{'...' if len(match_text) > 100 else ''}")
                else:
                    st.write("**Text:** None detected")
                
                # File info
                file_name = Path(match_path).name
                st.write(f"**File:** {file_name}")
    
    # Detailed results table
    st.subheader("📊 Detailed Results")
    
    results_data = []
    for i, (match_path, score, match_meta) in enumerate(zip(matches, scores, results.get('match_metadata', [{}]*len(matches)))):
        results_data.append({
            'Rank': i + 1,
            'Similarity %': f"{score * 100:.1f}%",
            'File': Path(match_path).name,
            'Text': match_meta.get('text', 'None')[:100] + ('...' if len(match_meta.get('text', '')) > 100 else ''),
            'Status': 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
        })
    
    st.dataframe(results_data, width='stretch')

def show_visual_comparison_ui():
    """Visual logo comparison UI"""
    st.markdown('<h1 class="main-header">🖼️ Visual Logo Comparison</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a step:",
        ["🏠 Home", "🔧 Initialize System", "📁 Upload PDFs", "🔍 Search Trademarks", "📊 System Status", "❓ Help"]
    )
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Advanced Trademark Logo Comparison System! 🎉
        
        This AI-powered system helps you find visually similar trademarks and logos using state-of-the-art computer vision and machine learning techniques.
        
        ### 🚀 Features:
        - **Multi-modal AI**: Combines Vision Transformer and EfficientNet models
        - **OCR Text Extraction**: Detects and matches text in logos
        - **Logo Region Detection**: Automatically crops and focuses on logo areas
        - **High-quality PDF Processing**: Extracts pages as high-resolution images
        - **Interactive Search**: Upload query images to find similar trademarks
        - **Similarity Scoring**: Get detailed similarity percentages and rankings
        
        ### 📋 How to Use:
        1. **Initialize System**: Load AI models (first time only)
        2. **Upload PDFs**: Upload your trademark PDF files
        3. **Search**: Upload query images to find similar trademarks
        4. **View Results**: See similarity scores and detailed analysis
        
        ### ⚠️ Important Notes:
        - First run may take a few minutes to download AI models
        - PDF processing time depends on number of pages
        - Best results with clear, high-contrast logos
        - Supported formats: PDF (for database), PNG/JPG/JPEG (for queries)
        """)
        
        # Quick start button
        if st.button("🚀 Quick Start - Initialize System", type="primary"):
            st.session_state.page = "🔧 Initialize System"
            st.rerun()
    
    elif page == "🔧 Initialize System":
        st.markdown('<h2 class="section-header">🔧 Initialize AI System</h2>', unsafe_allow_html=True)
        
        st.info("This step loads the AI models required for logo comparison. It may take a few minutes on first run.")
        
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_visual_system():
                st.success("✅ System initialized successfully!")
                st.session_state.index_built = False
                st.rerun()
            else:
                st.error("❌ Failed to initialize system!")
    
    elif page == "📁 Upload PDFs":
        st.markdown('<h2 class="section-header">📁 Upload Trademark PDFs</h2>', unsafe_allow_html=True)
        
        if st.session_state.trademark_system is None:
            st.error("❌ Please initialize the system first!")
            st.info("Go to 'Initialize System' tab to load the AI models.")
            return
        
        st.info("Upload one or more PDF files containing trademarks/logos. The system will extract each page as an image and build a searchable index.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files containing trademarks"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} PDF file(s) selected")
            
            # Show file details
            for i, file in enumerate(uploaded_files):
                st.write(f"**File {i+1}:** {file.name} ({file.size:,} bytes)")
            
            if st.button("🔄 Process PDFs and Build Index", type="primary"):
                with st.spinner("Processing PDFs and building search index..."):
                    try:
                        # Convert uploaded files to bytes
                        file_contents = [file.read() for file in uploaded_files]
                        file_names = [file.name for file in uploaded_files]
                        
                        # Process PDFs
                        image_paths = st.session_state.trademark_system.process_uploaded_pdfs(file_contents, file_names)
                        
                        if not image_paths:
                            st.error("❌ No images extracted from PDFs!")
                        else:
                            st.success(f"✅ Successfully extracted {len(image_paths)} images")
                            
                            # Build search index
                            st.session_state.trademark_system.build_index(image_paths)
                            st.session_state.index_built = True
                            
                            st.success("🎉 System ready for searches!")
                            
                            # Show sample extracted images
                            st.subheader("🖼️ Sample Extracted Images")
                            if len(image_paths) >= 3:
                                sample_paths = image_paths[:3]
                            else:
                                sample_paths = image_paths
                            
                            cols = st.columns(len(sample_paths))
                            for i, path in enumerate(sample_paths):
                                with cols[i]:
                                    try:
                                        img = Image.open(path)
                                        st.image(img, caption=f"Sample {i+1}", width='stretch')
                                    except Exception as e:
                                        st.error(f"Error loading sample {i+1}: {e}")
                            
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
    
    elif page == "🔍 Search Trademarks":
        st.markdown('<h2 class="section-header">🔍 Search for Similar Trademarks</h2>', unsafe_allow_html=True)
        
        if st.session_state.trademark_system is None:
            st.error("❌ Please initialize the system first!")
            return
        
        if not st.session_state.index_built:
            st.error("❌ Please upload and process PDFs first!")
            st.info("Go to 'Upload PDFs' tab to add trademark data.")
            return
        
        st.info("Upload an image (logo/trademark) to find similar ones in your database. Supported formats: PNG, JPG, JPEG")
        
        # Number of results
        num_results = st.slider("Number of results to show", 1, 10, 5)
        
        # Upload query image
        query_image = st.file_uploader(
            "Upload query image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a logo or trademark image to search for similar ones"
        )
        
        if query_image:
            # Store query image bytes in session state to avoid file path issues
            st.session_state.current_query_image = query_image.getvalue()
            
            # Show query image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Your Query Image")
                # Display the uploaded image directly from memory
                query_img = Image.open(query_image)
                st.image(query_img, width='stretch')
            
            with col2:
                st.subheader("🔍 Search Options")
                if st.button("🔍 Search for Similar Trademarks", type="primary"):
                    with st.spinner("Searching for similar trademarks..."):
                        try:
                            # Save uploaded image temporarily for processing
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{query_image.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(query_image.getvalue())
                                query_path = tmp_file.name
                            
                            # Perform search
                            results = st.session_state.trademark_system.search_similar(query_path, k=num_results)
                            
                            # Store results in session state to avoid file cleanup issues
                            st.session_state.last_search_results = results
                            
                            # Clean up temporary file after storing results
                            try:
                                os.unlink(query_path)
                            except:
                                pass
                                
                        except Exception as e:
                            st.error(f"❌ Search failed: {e}")
            
            # Display results if available
            if hasattr(st.session_state, 'last_search_results') and st.session_state.last_search_results:
                # Pass the query image bytes to avoid file path issues
                query_image_bytes = st.session_state.get('current_query_image')
                if query_image_bytes:
                    from io import BytesIO
                    show_search_results(st.session_state.last_search_results, max_results=num_results, query_image_bytes=BytesIO(query_image_bytes))
                else:
                    show_search_results(st.session_state.last_search_results, max_results=num_results)
    
    elif page == "📊 System Status":
        st.markdown('<h2 class="section-header">📊 System Status</h2>', unsafe_allow_html=True)
        
        if st.session_state.trademark_system is None:
            st.error("❌ System not initialized")
            return
        
        # System info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "✅ Initialized")
        
        with col2:
            device = st.session_state.trademark_system.device
            st.metric("Device", f"🖥️ {device.upper()}")
        
        with col3:
            if st.session_state.index_built:
                num_images = st.session_state.trademark_system.index.ntotal
                st.metric("Database Size", f"📁 {num_images} images")
            else:
                st.metric("Database Size", "📁 No data")
        
        # Model status
        st.subheader("🤖 AI Models Status")
        
        models_status = []
        if st.session_state.trademark_system.vit_model is not None:
            models_status.append("✅ Vision Transformer")
        if st.session_state.trademark_system.efficient_model is not None:
            models_status.append("✅ EfficientNet")
        if st.session_state.trademark_system.text_encoder is not None:
            models_status.append("✅ Text Encoder")
        if st.session_state.trademark_system.ocr_reader is not None:
            models_status.append("✅ OCR Reader")
        
        if models_status:
            for model in models_status:
                st.write(model)
        else:
            st.warning("⚠️ No models loaded")
        
        # Database info
        if st.session_state.index_built:
            st.subheader("📁 Database Information")
            
            metadata = st.session_state.trademark_system.metadata
            texts_found = sum(1 for m in metadata if m.get('text', ''))
            total_images = len(metadata)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Images", total_images)
            with col2:
                st.metric("Images with Text", f"{texts_found} ({texts_found/total_images*100:.1f}%)")
            
            # Sample extracted text
            if texts_found > 0:
                st.subheader("📝 Sample Extracted Text")
                sample_texts = [m['text'] for m in metadata[:5] if m.get('text', '')]
                for i, text in enumerate(sample_texts):
                    st.write(f"**{i+1}.** {text}")
    
    elif page == "❓ Help":
        st.markdown('<h2 class="section-header">❓ Help & Troubleshooting</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Quick Start Guide:
        1. **Initialize System** - Load AI models (first time only)
        2. **Upload PDFs** - Add your trademark PDF files
        3. **Search Trademarks** - Upload query images to find similar ones
        4. **Repeat searches** - Use different query images as needed
        
        ### ❌ Common Issues:
        
        **Problem:** "System not initialized"
        - **Solution:** Go to "Initialize System" tab and click the initialize button
        
        **Problem:** "No PDFs processed"
        - **Solution:** Go to "Upload PDFs" tab and upload your PDF files
        
        **Problem:** "No matches found"
        - **Solutions:**
          - Try uploading more PDF files to expand the database
          - Use clearer, higher-quality query images
          - Check if your query image is similar to logos in your PDFs
        
        **Problem:** Low similarity scores
        - **Solutions:**
          - The system is working correctly - low scores mean no similar logos found
          - Try different query images
          - Add more diverse trademark PDFs to your database
        
        ### 🎯 Tips for Best Results:
        - Use high-resolution, clear images
        - Logos should be prominently visible
        - Include both text-based and graphic-based trademarks
        - Upload multiple PDF files for a larger database
        
        ### 📊 Understanding Similarity Scores:
        - **90-100%**: Nearly identical (possible trademark conflict)
        - **70-89%**: Very similar (worth investigating)
        - **50-69%**: Somewhat similar (minor resemblance)
        - **<50%**: Different (no significant similarity)
        
        ### 🔧 Technical Details:
        - **AI Models**: Vision Transformer + EfficientNet + Text Encoder
        - **OCR**: EasyOCR for text extraction
        - **Search**: FAISS for fast similarity search
        - **Features**: Combines visual (80%) and text (20%) similarity
        """)

def show_main_interface():
    """Main interface with system selection"""
    # Main header
    st.markdown('<h1 class="main-header">🏷️ Trademark Analysis System</h1>', unsafe_allow_html=True)
    
    # System selection
    st.markdown("## 🚀 Choose Your Analysis System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="system-card">
            <h2>📄 Text-Based Extraction & Analysis</h2>
            <div class="feature-list">
                <p>✅ Extract structured data from PDFs/images</p>
                <p>✅ AI-powered text recognition (LLaMA models)</p>
                <p>✅ Fuzzy string matching for similarity</p>
                <p>✅ CSV database integration</p>
                <p>✅ Company & trademark information extraction</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Launch Text Analysis", key="text_analysis", type="primary"):
            st.session_state.current_system = "text_extraction"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="system-card">
            <h2>🖼️ Visual Logo Comparison</h2>
            <div class="feature-list">
                <p>✅ Computer vision logo matching</p>
                <p>✅ Vision Transformer + EfficientNet</p>
                <p>✅ OCR text extraction from logos</p>
                <p>✅ High-resolution PDF processing</p>
                <p>✅ Interactive similarity search</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Launch Visual Analysis", key="visual_analysis", type="primary"):
            st.session_state.current_system = "visual_comparison"
            st.rerun()
    
    # System information
    st.markdown("---")
    st.markdown("## ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Text Analysis")
        st.markdown("""
        - **AI Models**: LLaMA 3.3 70B, LLaMA 4 Scout
        - **Input**: PDF, JPG, PNG, BMP, TIFF
        - **Output**: Structured JSON data
        - **Use Case**: Document processing, data extraction
        """)
    
    with col2:
        st.markdown("### 🎨 Visual Analysis")
        st.markdown("""
        - **AI Models**: ViT, EfficientNet, SentenceTransformer
        - **Input**: PDF, PNG, JPG, JPEG
        - **Output**: Similarity scores, ranked results
        - **Use Case**: Logo matching, visual similarity
        """)
    
    with col3:
        st.markdown("### 🔄 Combined Workflow")
        st.markdown("""
        1. **Extract** data using Text Analysis
        2. **Compare** logos using Visual Analysis
        3. **Export** results for further processing
        4. **Integrate** with existing databases
        """)

def main():
    """Main application logic"""
    # Check if we're running from command line
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(
            description="Trademark Analysis System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --text-ui           # Run text extraction UI
  python main.py --visual-ui         # Run visual comparison UI
  python main.py --text-api          # Run text extraction API
  python main.py --all               # Show all options
            """
        )
        
        parser.add_argument(
            "--text-ui", 
            action="store_true", 
            help="Run text extraction Streamlit UI"
        )
        parser.add_argument(
            "--visual-ui", 
            action="store_true", 
            help="Run visual comparison Streamlit UI"
        )
        parser.add_argument(
            "--text-api", 
            action="store_true", 
            help="Run text extraction FastAPI server"
        )
        parser.add_argument(
            "--all", 
            action="store_true", 
            help="Show all available options"
        )
        
        args = parser.parse_args()
        
        if args.all:
            print("🏷️ Trademark Analysis System")
            print("=" * 50)
            print("\nAvailable Options:")
            print("1. Text Extraction UI (Streamlit)")
            print("   Command: python main.py --text-ui")
            print("   URL: http://localhost:8501")
            print("\n2. Visual Comparison UI (Streamlit)")
            print("   Command: python main.py --visual-ui")
            print("   URL: http://localhost:8501")
            print("\n3. Text Extraction API (FastAPI)")
            print("   Command: python main.py --text-api")
            print("   URL: http://localhost:8000")
            print("\n4. Default (Main Interface)")
            print("   Command: python main.py")
            print("   URL: http://localhost:8501")
            return
        
        if args.text_api:
            # Run FastAPI server
            print("🚀 Starting Text Extraction API Server...")
            print("📄 API Documentation: http://localhost:8000/docs")
            print("🔗 Health check: http://localhost:8000/health")
            
            import uvicorn
            from text_extraction.controllers.trademark_controller import trademark_router
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            
            # Create FastAPI application
            app = FastAPI(
                title="Trademark Text Extraction API",
                description="API for extracting trademark data from PDF files and images",
                version="1.0.0",
                docs_url="/docs",
                redoc_url="/redoc"
            )
            
            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Include the trademark router
            app.include_router(trademark_router, prefix="/api/v1", tags=["trademark"])
            
            @app.get("/")
            async def root():
                return {
                    "message": "Trademark Text Extraction API",
                    "version": "1.0.0",
                    "docs": "/docs",
                    "health": "/api/v1/trademark/health"
                }
            
            @app.get("/health")
            async def health_check():
                return {
                    "status": "healthy",
                    "message": "Trademark Text Extraction API is running"
                }
            
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info"
            )
            return
        
        if args.text_ui:
            st.session_state.current_system = "text_extraction"
        elif args.visual_ui:
            st.session_state.current_system = "visual_comparison"
    
    # Streamlit UI mode
    if st.session_state.current_system == "text_extraction":
        if st.button("🏠 Back to System Selection", key="back_to_selection"):
            st.session_state.current_system = None
            st.rerun()
        show_text_extraction_ui()
    elif st.session_state.current_system == "visual_comparison":
        if st.button("🏠 Back to System Selection", key="back_to_selection"):
            st.session_state.current_system = None
            st.rerun()
        show_visual_comparison_ui()
    else:
        show_main_interface()

if __name__ == "__main__":
    main()
