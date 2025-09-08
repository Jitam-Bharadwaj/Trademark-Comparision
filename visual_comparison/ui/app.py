import streamlit as st
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from visual_comparison.DAL_files.trademark_search import InteractiveTrademarkSearch

# Page configuration
st.set_page_config(
    page_title="🏢 Trademark Logo Comparison System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trademark_system' not in st.session_state:
    st.session_state.trademark_system = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = None
if 'current_query_image' not in st.session_state:
    st.session_state.current_query_image = None

def initialize_system():
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

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏢 Advanced Trademark Logo Comparison System</h1>', unsafe_allow_html=True)
    
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
            if initialize_system():
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

if __name__ == "__main__":
    main()
