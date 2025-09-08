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

# Import your existing classes
from text_extraction.DAL_files.trademark_dal import TrademarkExtractor
from text_extraction.DAL_files.trademark_comparison import TrademarkComparator
from text_extraction.DAL_files.csv_manager import TrademarkCSVManager

# Configure Streamlit page
st.set_page_config(
    page_title="Trademark Extraction & Comparison Tool",
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
    .similarity-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;s
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

# Initialize classes
@st.cache_resource
def initialize_components():
    """Initialize trademark processing components"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "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")
        extractor = TrademarkExtractor(groq_api_key=groq_api_key)
        comparator = TrademarkComparator()
        csv_manager = TrademarkCSVManager()
        return extractor, comparator, csv_manager
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None

extractor, comparator, csv_manager = initialize_components()

# Main title
st.markdown('<h1 class="main-header">🏷️ Trademark Extraction & Comparison Tool</h1>', unsafe_allow_html=True)

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
    
    # Database stats
    if st.session_state.csv_loaded:
        st.subheader("📈 Database Statistics")
        try:
            stats = csv_manager.get_csv_stats()
            st.metric("Total Trademarks", stats.get('total_trademarks', 0))
            
            # Show CSV structure analysis
            with st.expander("🔍 CSV Structure Analysis"):
                # Load and analyze the CSV
                csv_data = csv_manager.load_trademarks_from_csv()
                if csv_data:
                    # Show column analysis
                    st.write("**Available Columns:**")
                    sample_row = csv_data[0] if csv_data else {}
                    
                    for col in csv_manager.fieldnames:
                        non_empty_count = sum(1 for row in csv_data if row.get(col, '').strip())
                        total_count = len(csv_data)
                        percentage = (non_empty_count / total_count * 100) if total_count > 0 else 0
                        
                        if percentage > 0:
                            st.write(f"✅ **{col}**: {non_empty_count}/{total_count} ({percentage:.1f}%) have data")
                        else:
                            st.write(f"❌ **{col}**: No data found")
                    
                    # Show sample data
                    st.write("**Sample Rows (first 3):**")
                    for i, row in enumerate(csv_data[:3]):
                        st.write(f"**Row {i+1}:**")
                        for key, value in row.items():
                            if value and value.strip():
                                st.text(f"  {key}: '{value}'")
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"Error loading CSV stats: {str(e)}")

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
                    # Add a summary box showing what will be compared
                    st.markdown("""
                    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h5 style="margin: 0; color: #1976d2;">🔍 Names for Comparison:</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.write(f"**Company Name:** `{trademark.get('name', 'N/A')}`")
                    with comp_col2:
                        st.write(f"**Trademark Text:** `{trademark.get('text_in_logo', 'N/A')}`")
                    
                    st.markdown("---")
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
                        
                        # Add a summary table first
                        st.markdown("**📊 Quick Comparison Overview:**")
                        summary_data = []
                        for k, match in enumerate(result['matches']):
                            # Determine what was actually compared for trademark
                            csv_trademark_display = match['existing_trademark'].get('Trademark', 'N/A')
                            if match.get('fallback_used', False):
                                csv_trademark_display = f"{csv_trademark_display} → {match['existing_trademark'].get('Client / Applicant', 'N/A')} (fallback)"
                            
                            summary_data.append({
                                "Match #": k + 1,
                                "PDF Company": result['new_trademark']['name'][:30] + "..." if len(result['new_trademark']['name']) > 30 else result['new_trademark']['name'],
                                "CSV Company": match['existing_trademark'].get('Client / Applicant', 'N/A')[:30] + "..." if len(match['existing_trademark'].get('Client / Applicant', '')) > 30 else match['existing_trademark'].get('Client / Applicant', 'N/A'),
                                "PDF Trademark": result['new_trademark']['trademark'][:25] + "..." if len(result['new_trademark']['trademark']) > 25 else result['new_trademark']['trademark'],
                                "CSV Trademark": csv_trademark_display[:40] + "..." if len(csv_trademark_display) > 40 else csv_trademark_display,
                                "Similarity": f"{match['similarity_score']:.1f}%",
                                "Type": "🔄 Fallback" if match.get('fallback_used', False) else "✅ Normal"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        st.markdown("---")
                        
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
                                
                                # Match details - showing what's being compared
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
                                    
                                    # Show fallback information
                                    if match.get('fallback_used', False):
                                        st.info("🔄 **Fallback Used**: CSV Trademark column was empty, so Client/Applicant name was used for trademark comparison")
                                    
                                    # Detailed scores
                                    detailed = match['detailed_scores']
                                    name_score = detailed['name_comparison']['overall']
                                    logo_score = detailed['logo_comparison']['overall']
                                    
                                    st.write(f"**Name Similarity:** {name_score:.1f}%")
                                    st.write(f"**Logo Similarity:** {logo_score:.1f}%")
                                
                                # Add a clear comparison section
                                st.markdown("---")
                                st.markdown("**🔍 What's Being Compared:**")
                                comp_col1, comp_col2 = st.columns(2)
                                
                                with comp_col1:
                                    st.markdown("**Company Names:**")
                                    pdf_name = new_trademark.get('name', 'N/A')
                                    csv_name = existing.get('Client / Applicant', 'N/A')
                                    st.write(f"📄 PDF: `{pdf_name}`")
                                    st.write(f"🗃️ CSV: `{csv_name}`")
                                    st.write(f"📈 Score: **{name_score:.1f}%**")
                                
                                with comp_col2:
                                    st.markdown("**Trademark Text:**")
                                    pdf_trademark = new_trademark.get('trademark', 'N/A')
                                    csv_trademark_orig = existing.get('Trademark', 'N/A')
                                    
                                    # Show what's actually being compared (considering fallback)
                                    if match.get('fallback_used', False):
                                        csv_trademark_compared = existing.get('Client / Applicant', 'N/A')
                                        st.write(f"📄 PDF: `{pdf_trademark}`")
                                        st.write(f"🗃️ CSV: `{csv_trademark_orig}` (empty)")
                                        st.write(f"🔄 **Actually Compared Against**: `{csv_trademark_compared}` (Client/Applicant)")
                                    else:
                                        csv_trademark_compared = csv_trademark_orig
                                        st.write(f"📄 PDF: `{pdf_trademark}`")
                                        st.write(f"🗃️ CSV: `{csv_trademark_compared}`")
                                    
                                    st.write(f"📈 Score: **{logo_score:.1f}%**")
                                
                                # Show normalized comparison
                                with st.expander("🔍 Normalized Text Comparison (After Processing)"):
                                    st.markdown("*This shows how the text appears after normalization for comparison:*")
                                    
                                    # Debug: Show raw values first
                                    st.markdown("**🔍 Raw Values from Data:**")
                                    debug_col1, debug_col2 = st.columns(2)
                                    with debug_col1:
                                        st.write("**PDF Raw Values:**")
                                        st.code(f"name: '{pdf_name}'")
                                        st.code(f"text_in_logo: '{pdf_trademark}'")
                                    
                                    with debug_col2:
                                        st.write("**CSV Raw Values:**")
                                        st.code(f"Client / Applicant: '{csv_name}'")
                                        st.code(f"Trademark: '{csv_trademark_orig}'")
                                        
                                        # Show fallback logic
                                        if match.get('fallback_used', False):
                                            st.warning("🔄 **Fallback Applied**: Trademark column empty, using Client/Applicant for comparison")
                                            st.code(f"Compared Against: '{csv_trademark_compared}' (Client/Applicant)")
                                        
                                        # Show all CSV columns for this row
                                        st.write("**All CSV columns for this row:**")
                                        for key, value in existing.items():
                                            if value and value.strip():  # Only show non-empty values
                                                st.text(f"{key}: '{value}'")
                                    
                                    st.markdown("---")
                                    
                                    # Use comparator to show normalized versions
                                    norm_pdf_name = comparator.normalize_name(pdf_name) if pdf_name and pdf_name != 'N/A' else 'N/A'
                                    norm_csv_name = comparator.normalize_name(csv_name) if csv_name and csv_name != 'N/A' else 'N/A'
                                    norm_pdf_trademark = comparator.normalize_name(pdf_trademark) if pdf_trademark and pdf_trademark != 'N/A' else 'N/A'
                                    norm_csv_trademark = comparator.normalize_name(csv_trademark_compared) if csv_trademark_compared and csv_trademark_compared != 'N/A' else 'N/A'
                                    
                                    norm_col1, norm_col2 = st.columns(2)
                                    with norm_col1:
                                        st.write("**Normalized Company Names:**")
                                        st.code(f"PDF: {norm_pdf_name}")
                                        st.code(f"CSV: {norm_csv_name}")
                                    
                                    with norm_col2:
                                        st.write("**Normalized Trademark Text:**")
                                        st.code(f"PDF: {norm_pdf_trademark}")
                                        
                                        if match.get('fallback_used', False):
                                            st.code(f"CSV (Fallback): {norm_csv_trademark}")
                                            st.info("✅ **Fallback Success**: Used Client/Applicant name for comparison since Trademark column was empty")
                                        else:
                                            st.code(f"CSV: {norm_csv_trademark}")
                                        
                                        # Explain why CSV might be N/A
                                        if csv_trademark_orig == 'N/A' or not csv_trademark_orig or not csv_trademark_orig.strip():
                                            if not match.get('fallback_used', False):
                                                st.warning("⚠️ CSV trademark text is empty/missing. Consider using the fallback logic to compare against Client/Applicant name.")
                                
                                # Detailed analysis in expander
                                with st.expander(f"📊 Detailed Analysis for Match {j+1}"):
                                    # Create similarity charts
                                    fig = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Name Comparison', 'Logo Comparison'),
                                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                                    )
                                    
                                    # Name comparison scores
                                    name_scores = detailed['name_comparison']['fuzzy']
                                    fig.add_trace(
                                        go.Bar(
                                            x=list(name_scores.keys()),
                                            y=list(name_scores.values()),
                                            name="Name Scores",
                                            marker_color='lightblue'
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # Logo comparison scores
                                    logo_scores = detailed['logo_comparison']['fuzzy']
                                    fig.add_trace(
                                        go.Bar(
                                            x=list(logo_scores.keys()),
                                            y=list(logo_scores.values()),
                                            name="Logo Scores",
                                            marker_color='lightcoral'
                                        ),
                                        row=1, col=2
                                    )
                                    
                                    fig.update_layout(
                                        height=400,
                                        showlegend=False,
                                        title_text=f"Detailed Similarity Analysis - Match {j+1}"
                                    )
                                    fig.update_yaxes(range=[0, 100])
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No similar trademarks found in the database above the similarity threshold.")
                    
                    st.markdown("---")
            
            else:
                st.info("💡 Upload a CSV database to enable similarity comparison.")
            
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")

# Additional features section
if st.session_state.extracted_trademarks or st.session_state.comparison_results:
    st.header("📊 Analysis Dashboard")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Summary Statistics", "🗂️ Export Data", "⚙️ Settings"])
    
    with tab1:
        if st.session_state.comparison_results:
            # Overall statistics
            total_matches = sum([result['similar_trademarks_found'] for result in st.session_state.comparison_results])
            total_extracted = len(st.session_state.extracted_trademarks)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trademarks Extracted", total_extracted)
            with col2:
                st.metric("Total Matches Found", total_matches)
            with col3:
                if total_extracted > 0:
                    conflict_rate = (len([r for r in st.session_state.comparison_results if r['similar_trademarks_found'] > 0]) / total_extracted) * 100
                    st.metric("Conflict Rate", f"{conflict_rate:.1f}%")
            
            # Similarity distribution chart
            if total_matches > 0:
                all_scores = []
                for result in st.session_state.comparison_results:
                    for match in result['matches']:
                        all_scores.append(match['similarity_score'])
                
                if all_scores:
                    fig = px.histogram(
                        x=all_scores,
                        nbins=20,
                        title="Distribution of Similarity Scores",
                        labels={'x': 'Similarity Score (%)', 'y': 'Count'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📤 Export Results")
        
        if st.session_state.extracted_trademarks:
            # Convert to DataFrame for export
            export_data = []
            for i, trademark in enumerate(st.session_state.extracted_trademarks):
                row = trademark.copy()
                row['extraction_order'] = i + 1
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"extracted_trademarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(st.session_state.extracted_trademarks, indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_data,
                    file_name=f"extracted_trademarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        if st.session_state.comparison_results:
            # Export comparison results
            comparison_json = json.dumps(st.session_state.comparison_results, indent=2)
            st.download_button(
                label="🔍 Download Comparison Results (JSON)",
                data=comparison_json,
                file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with tab3:
        st.subheader("🛠️ Advanced Settings")
        
        # Clear session state
        if st.button("🗑️ Clear All Results"):
            st.session_state.extracted_trademarks = []
            st.session_state.comparison_results = []
            st.rerun()
        
        # API Key management
        st.markdown("**🔑 API Configuration**")
        current_key = os.getenv("GROQ_API_KEY", "")
        if current_key:
            st.success("✅ GROQ API Key is configured")
        else:
            st.warning("⚠️ GROQ API Key not found in environment variables")
        
        # CSV file info
        if os.path.exists(csv_manager.csv_file_path):
            st.markdown("**📁 CSV Database File**")
            st.code(f"Path: {csv_manager.csv_file_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏷️ Trademark Extraction & Comparison Tool | Built with Streamlit</p>
    <p>Upload PDF files to extract trademark information and find similar trademarks in your database.</p>
</div>
""", unsafe_allow_html=True)