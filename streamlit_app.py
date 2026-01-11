import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from io import BytesIO
import requests

# Page configuration
st.set_page_config(
    page_title="Topic Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path_or_url, file_type='auto', use_chunking=False, chunk_size=100000):
    """
    Load data from CSV or Parquet file (local path or URL) into a pandas DataFrame.
    Optimized for large files with optional chunking.
    
    Parameters:
    - file_path_or_url: Path to local file or URL
    - file_type: 'auto', 'csv', or 'parquet'
    - use_chunking: If True, load in chunks (for very large files)
    - chunk_size: Number of rows per chunk
    """
    try:
        # Auto-detect file type
        if file_type == 'auto':
            if file_path_or_url.endswith('.parquet') or file_path_or_url.endswith('.pq'):
                file_type = 'parquet'
            elif file_path_or_url.endswith('.csv'):
                file_type = 'csv'
            else:
                # Try to detect from URL or check first bytes
                file_type = 'csv'  # Default to CSV
        
        # Check if it's a URL
        is_url = file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://')
        
        if file_type == 'parquet':
            if is_url:
                # Download parquet file
                response = requests.get(file_path_or_url, stream=True)
                response.raise_for_status()
                parquet_file = BytesIO(response.content)
                df = pd.read_parquet(parquet_file)
            else:
                # Read local parquet file
                if use_chunking:
                    # For very large files, read in chunks
                    parquet_file = pq.ParquetFile(file_path_or_url)
                    chunks = []
                    for batch in parquet_file.iter_batches(batch_size=chunk_size):
                        chunks.append(batch.to_pandas())
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_parquet(file_path_or_url)
        else:  # CSV
            if is_url:
                # For CSV from URL, use pandas directly
                if use_chunking:
                    chunks = []
                    for chunk in pd.read_csv(file_path_or_url, chunksize=chunk_size):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path_or_url)
            else:
                # Local CSV file
                if use_chunking:
                    chunks = []
                    for chunk in pd.read_csv(file_path_or_url, chunksize=chunk_size):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path_or_url)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.error(f"File type: {file_type}, URL: {is_url if 'is_url' in locals() else 'unknown'}")
        return None

def aggregate_by_topic(df):
    """Aggregate data by topic, summing raw_conversations"""
    aggregated = df.groupby('topic', as_index=False).agg({
        'raw_conversations': 'sum',
        'raw_messages': 'sum',
        'raw_users': 'sum'
    }).sort_values('raw_conversations', ascending=False)
    
    return aggregated

def create_top_topics_plot(df, top_n=10, title_suffix=""):
    """Create a bar plot of top N topics by raw_conversations"""
    top_topics = df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_topics)), top_topics['raw_conversations'], 
                   color=sns.color_palette("viridis", len(top_topics)))
    
    # Customize the plot
    ax.set_yticks(range(len(top_topics)))
    ax.set_yticklabels(top_topics['topic'], fontsize=10)
    ax.set_xlabel('Raw Conversations', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Topics by Raw Conversations{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_topics.iterrows()):
        value = row['raw_conversations']
        ax.text(value, i, f' {value:,}', va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Topic Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for data source configuration
    st.sidebar.header("📁 Data Source")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["GitHub CSV (Default)", "GitHub Parquet", "Local File", "Custom URL"],
        index=0
    )
    
    # File type selection
    file_type = st.sidebar.selectbox("File Type", ["auto", "csv", "parquet"], index=0)
    
    # Chunking option for large files
    use_chunking = st.sidebar.checkbox("Use Chunking (for large files >1GB)", value=False)
    chunk_size = None
    if use_chunking:
        chunk_size = st.sidebar.number_input("Chunk Size (rows)", min_value=10000, max_value=1000000, value=100000, step=10000)
    
    # Determine data source URL/path
    if data_source == "GitHub CSV (Default)":
        data_url = "https://raw.githubusercontent.com/litancherikover/Topic_analysis/main/Prompts.csv"
        file_type = "csv"
    elif data_source == "GitHub Parquet":
        data_url = st.sidebar.text_input(
            "Parquet File URL",
            value="https://raw.githubusercontent.com/litancherikover/Topic_analysis/main/Prompts.parquet"
        )
        file_type = "parquet"
    elif data_source == "Local File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Parquet file", type=['csv', 'parquet', 'pq'])
        if uploaded_file:
            data_url = uploaded_file
            file_type = "auto"
        else:
            st.warning("Please upload a file")
            st.stop()
    else:  # Custom URL
        data_url = st.sidebar.text_input("Enter file URL", value="")
        if not data_url:
            st.warning("Please enter a file URL")
            st.stop()
    
    # Load data
    loading_message = "Loading data..."
    if use_chunking:
        loading_message += f" (using chunks of {chunk_size:,} rows)"
    
    with st.spinner(loading_message):
        if data_source == "Local File" and uploaded_file:
            # For uploaded files, read directly
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        else:
            df = load_data(data_url, file_type=file_type, use_chunking=use_chunking, chunk_size=chunk_size)
    
    if df is None:
        st.stop()
    
    # Show available columns for debugging
    st.sidebar.divider()
    with st.sidebar.expander("📋 Available Columns"):
        st.write(df.columns.tolist())
    
    # Column mapping - detect or use defaults
    # Try to find the right columns automatically
    col_level1 = 'level1' if 'level1' in df.columns else ('category' if 'category' in df.columns else None)
    col_level2 = 'level2' if 'level2' in df.columns else ('subcategory' if 'subcategory' in df.columns else None)
    col_topic = 'topic' if 'topic' in df.columns else ('prompt' if 'prompt' in df.columns else None)
    col_date = 'date' if 'date' in df.columns else None
    col_conversations = 'raw_conversations' if 'raw_conversations' in df.columns else ('conversations' if 'conversations' in df.columns else None)
    col_users = 'raw_users' if 'raw_users' in df.columns else ('users' if 'users' in df.columns else None)
    col_messages = 'raw_messages' if 'raw_messages' in df.columns else ('messages' if 'messages' in df.columns else None)
    
    # Let user select columns if auto-detection fails
    st.sidebar.header("⚙️ Column Mapping")
    all_columns = ['None'] + df.columns.tolist()
    
    col_level1 = st.sidebar.selectbox("Category Column (Level 1)", all_columns, index=all_columns.index(col_level1) if col_level1 in all_columns else 0)
    col_level2 = st.sidebar.selectbox("Subcategory Column (Level 2)", all_columns, index=all_columns.index(col_level2) if col_level2 in all_columns else 0)
    col_topic = st.sidebar.selectbox("Topic/Prompt Column", all_columns, index=all_columns.index(col_topic) if col_topic in all_columns else 0)
    col_conversations = st.sidebar.selectbox("Conversations Column", all_columns, index=all_columns.index(col_conversations) if col_conversations in all_columns else 0)
    
    # Sidebar filters
    st.sidebar.divider()
    st.sidebar.header("🔍 Filters")
    
    # Date filter (if date column exists)
    start_date = end_date = None
    if col_date and col_date in df.columns:
        df[col_date] = pd.to_datetime(df[col_date])
        min_date = df[col_date].min().date()
        max_date = df[col_date].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
    
    # Level1 filter
    selected_level1 = 'All'
    if col_level1 and col_level1 != 'None' and col_level1 in df.columns:
        level1_options = ['All'] + sorted(df[col_level1].dropna().unique().tolist())
        selected_level1 = st.sidebar.selectbox("Category (Level 1)", level1_options)
    
    # Level2 filter (dependent on level1)
    selected_level2 = 'All'
    if col_level2 and col_level2 != 'None' and col_level2 in df.columns:
        if selected_level1 == 'All':
            level2_options = ['All'] + sorted(df[col_level2].dropna().unique().tolist())
        else:
            level2_options = ['All'] + sorted(df[df[col_level1] == selected_level1][col_level2].dropna().unique().tolist())
        selected_level2 = st.sidebar.selectbox("Subcategory (Level 2)", level2_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if col_date and col_date in df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df[col_date].dt.date >= start_date) & 
            (filtered_df[col_date].dt.date <= end_date)
        ]
    
    # Level1 filter
    if selected_level1 != 'All' and col_level1 and col_level1 != 'None':
        filtered_df = filtered_df[filtered_df[col_level1] == selected_level1]
    
    # Level2 filter
    if selected_level2 != 'All' and col_level2 and col_level2 != 'None':
        filtered_df = filtered_df[filtered_df[col_level2] == selected_level2]
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(filtered_df):,}")
    
    with col2:
        if col_topic and col_topic != 'None' and col_topic in filtered_df.columns:
            st.metric("Unique Topics", f"{filtered_df[col_topic].nunique():,}")
        else:
            st.metric("Unique Topics", "N/A")
    
    with col3:
        if col_conversations and col_conversations != 'None' and col_conversations in filtered_df.columns:
            st.metric("Total Conversations", f"{filtered_df[col_conversations].sum():,}")
        else:
            st.metric("Total Conversations", "N/A")
    
    with col4:
        if col_users and col_users != 'None' and col_users in filtered_df.columns:
            st.metric("Total Users", f"{filtered_df[col_users].sum():,}")
        else:
            st.metric("Total Users", "N/A")
    
    st.divider()
    
    # Aggregated data
    if len(filtered_df) > 0:
        # Aggregate by topic column
        if col_topic and col_topic != 'None' and col_topic in filtered_df.columns:
            # Build aggregation dict dynamically
            agg_dict = {}
            if col_conversations and col_conversations != 'None' and col_conversations in filtered_df.columns:
                agg_dict[col_conversations] = 'sum'
            if col_messages and col_messages != 'None' and col_messages in filtered_df.columns:
                agg_dict[col_messages] = 'sum'
            if col_users and col_users != 'None' and col_users in filtered_df.columns:
                agg_dict[col_users] = 'sum'
            
            if agg_dict:
                aggregated = filtered_df.groupby(col_topic, as_index=False).agg(agg_dict)
                sort_col = col_conversations if col_conversations and col_conversations != 'None' else list(agg_dict.keys())[0]
                aggregated = aggregated.sort_values(sort_col, ascending=False)
            else:
                # Just count occurrences
                aggregated = filtered_df[col_topic].value_counts().reset_index()
                aggregated.columns = [col_topic, 'count']
                sort_col = 'count'
        else:
            st.warning("⚠️ Please select a Topic/Prompt column to aggregate data.")
            st.stop()
        
        # Top 10 topics visualization
        metric_name = col_conversations if col_conversations and col_conversations != 'None' else sort_col
        st.subheader(f"📈 Top 10 Topics by {metric_name}")
        
        # Create title suffix for filters
        title_parts = []
        if selected_level1 != 'All':
            title_parts.append(f"category: {selected_level1}")
        if selected_level2 != 'All':
            title_parts.append(f"subcategory: {selected_level2}")
        if start_date and end_date:
            if start_date != end_date:
                title_parts.append(f"dates: {start_date} to {end_date}")
            else:
                title_parts.append(f"date: {start_date}")
        
        title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
        
        # Create and display plot
        top_topics = aggregated.head(10)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(range(len(top_topics)), top_topics[sort_col], 
                       color=sns.color_palette("viridis", len(top_topics)))
        
        ax.set_yticks(range(len(top_topics)))
        ax.set_yticklabels(top_topics[col_topic], fontsize=10)
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'Top 10 Topics by {metric_name}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
        
        for i, val in enumerate(top_topics[sort_col]):
            ax.text(val, i, f' {val:,.0f}', va='center', fontsize=9, fontweight='bold')
        
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        # Display top 10 table
        st.subheader("📋 Top 10 Topics Table")
        
        display_df = aggregated.head(10).copy()
        display_df.index = range(1, len(display_df) + 1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button for filtered data
        st.subheader("💾 Download Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_topics_{selected_level1}_{selected_level2}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ No data found with the selected filters. Please adjust your filter criteria.")
    
    # Sidebar info
    st.sidebar.divider()
    st.sidebar.info("""
    **How to use:**
    1. Select date range (if multiple dates available)
    2. Choose Level 1 category
    3. Choose Level 2 category (filtered by Level 1)
    4. View top 10 topics visualization
    5. Download filtered data if needed
    """)

if __name__ == "__main__":
    main()
