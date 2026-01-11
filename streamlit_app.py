import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from io import BytesIO
import requests
import plotly.express as px
import plotly.graph_objects as go

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
    
    # Default data source - load from GitHub
    data_url = "https://raw.githubusercontent.com/litancherikover/Topic_analysis/main/Prompts%20(1).csv"
    file_type = "csv"
    use_chunking = False
    chunk_size = None
    uploaded_file = None
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(data_url, file_type=file_type, use_chunking=use_chunking, chunk_size=chunk_size)
    
    if df is None:
        st.stop()
    
    # Auto-detect columns
    col_level1 = None
    col_level2 = None
    col_topic = None
    col_date = None
    col_conversations = None
    col_users = None
    col_messages = None
    
    # Try to find the right columns automatically
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['level1', 'category', 'l1', 'cat']:
            col_level1 = col
        elif col_lower in ['level2', 'subcategory', 'l2', 'subcat', 'sub_category']:
            col_level2 = col
        elif col_lower in ['topic', 'prompt', 'query', 'search_term']:
            col_topic = col
        elif col_lower in ['date', 'datetime', 'timestamp']:
            col_date = col
        elif col_lower in ['raw_conversations', 'conversations', 'conv', 'count']:
            col_conversations = col
        elif col_lower in ['raw_users', 'users', 'user_count']:
            col_users = col
        elif col_lower in ['raw_messages', 'messages', 'msg_count']:
            col_messages = col
    
    # Column mapping in expander (for advanced users)
    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Column Mapping (click to configure)"):
        st.caption("Select which columns to use for filtering and analysis")
        all_columns = ['None'] + df.columns.tolist()
        
        col_level1 = st.selectbox("Category Column", all_columns, 
                                   index=all_columns.index(col_level1) if col_level1 in all_columns else 0,
                                   key="col_level1")
        col_level2 = st.selectbox("Subcategory Column", all_columns, 
                                   index=all_columns.index(col_level2) if col_level2 in all_columns else 0,
                                   key="col_level2")
        col_topic = st.selectbox("Topic/Prompt Column", all_columns, 
                                  index=all_columns.index(col_topic) if col_topic in all_columns else 0,
                                  key="col_topic")
        col_conversations = st.selectbox("Metric Column (for ranking)", all_columns, 
                                          index=all_columns.index(col_conversations) if col_conversations in all_columns else 0,
                                          key="col_conv")
        
        st.caption(f"Available columns: {', '.join(df.columns.tolist())}")
    
    # ============================================
    # MAIN FILTERS SECTION
    # ============================================
    st.sidebar.header("🔍 Filter Data")
    
    # Category filter (Level 1)
    selected_level1 = 'All'
    if col_level1 and col_level1 != 'None' and col_level1 in df.columns:
        level1_values = sorted([str(x) for x in df[col_level1].dropna().unique().tolist()])
        level1_options = ['All'] + level1_values
        selected_level1 = st.sidebar.selectbox(
            f"📁 Category ({col_level1})", 
            level1_options,
            help=f"Filter by {col_level1} - {len(level1_values)} unique values"
        )
    else:
        st.sidebar.warning("⚠️ No category column detected. Configure in Column Mapping above.")
    
    # Subcategory filter (Level 2) - dependent on category
    selected_level2 = 'All'
    if col_level2 and col_level2 != 'None' and col_level2 in df.columns:
        if selected_level1 == 'All':
            level2_values = sorted([str(x) for x in df[col_level2].dropna().unique().tolist()])
        else:
            level2_values = sorted([str(x) for x in df[df[col_level1] == selected_level1][col_level2].dropna().unique().tolist()])
        level2_options = ['All'] + level2_values
        selected_level2 = st.sidebar.selectbox(
            f"📂 Subcategory ({col_level2})", 
            level2_options,
            help=f"Filter by {col_level2} - {len(level2_values)} unique values"
        )
    
    # Date filter (if date column exists)
    start_date = end_date = None
    if col_date and col_date in df.columns:
        try:
            df[col_date] = pd.to_datetime(df[col_date])
            min_date = df[col_date].min().date()
            max_date = df[col_date].max().date()
            
            date_range = st.sidebar.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range
        except Exception:
            pass  # Skip date filter if conversion fails
    
    # ============================================
    # APPLY FILTERS
    # ============================================
    filtered_df = df.copy()
    
    # Apply category filter
    if selected_level1 != 'All' and col_level1 and col_level1 != 'None':
        filtered_df = filtered_df[filtered_df[col_level1].astype(str) == selected_level1]
    
    # Apply subcategory filter
    if selected_level2 != 'All' and col_level2 and col_level2 != 'None':
        filtered_df = filtered_df[filtered_df[col_level2].astype(str) == selected_level2]
    
    # Apply date filter
    if col_date and col_date in df.columns and start_date and end_date:
        try:
            filtered_df = filtered_df[
                (filtered_df[col_date].dt.date >= start_date) & 
                (filtered_df[col_date].dt.date <= end_date)
            ]
        except Exception:
            pass
    
    # Show filter summary
    st.sidebar.divider()
    st.sidebar.success(f"✅ Showing {len(filtered_df):,} of {len(df):,} rows")
    
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
        
        # Create filtered version for charts (exclude "all" topic as it skews visualization)
        aggregated_for_charts = aggregated[~aggregated[col_topic].astype(str).str.lower().isin(['all', 'total', 'overall'])].copy()
        
        # Create and display plot (using filtered data without "all")
        top_topics = aggregated_for_charts.head(10)
        
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
        
        # ============================================
        # TOP 10 TRENDS OVER TIME
        # ============================================
        if col_date and col_date in filtered_df.columns:
            st.subheader(f"📊 Top 10 Trends Over Time")
            
            # Get top 10 topics (excluding "all" for charts)
            top_10_topics = aggregated_for_charts.head(10)[col_topic].tolist()
            
            # Filter data for top 10 topics only
            trends_df = filtered_df[filtered_df[col_topic].isin(top_10_topics)].copy()
            
            if len(trends_df) > 0 and col_conversations and col_conversations != 'None':
                # Convert datetime to date only (remove time component)
                trends_df['date_only'] = pd.to_datetime(trends_df[col_date]).dt.date
                
                # Aggregate by date (not datetime) and topic
                trends_agg = trends_df.groupby(['date_only', col_topic], as_index=False).agg({
                    col_conversations: 'sum'
                })
                
                # Sort by date
                trends_agg = trends_agg.sort_values('date_only')
                trends_agg['date_only'] = pd.to_datetime(trends_agg['date_only'])
                
                # Create interactive Plotly line chart
                fig2 = px.line(
                    trends_agg,
                    x='date_only',
                    y=col_conversations,
                    color=col_topic,
                    title=f'Top 10 Topics Trend Over Time{title_suffix}',
                    labels={
                        'date_only': 'Date',
                        col_conversations: col_conversations,
                        col_topic: 'Topic'
                    },
                    markers=True
                )
                
                # Update layout for better appearance
                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title=col_conversations,
                    legend_title="Topic",
                    hovermode='x unified',
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.4,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10)
                    )
                )
                
                # Format x-axis dates
                fig2.update_xaxes(
                    tickformat="%Y-%m-%d",
                    tickangle=45
                )
                
                # Add hover template
                fig2.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Conversations: %{y:,.0f}<extra></extra>'
                )
                
                # Display interactive chart
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Need date and conversations columns to show trends over time.")
        else:
            st.info("ℹ️ No date column available to show trends over time.")
        
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
