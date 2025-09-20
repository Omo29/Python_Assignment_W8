import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center;}
    .section-header {font-size: 2rem; color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: 0.5rem;}
    .info-text {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
st.write("Interactive exploration of COVID-19 research papers metadata")

# Part 1: Data Loading and Basic Exploration
@st.cache_data
def load_data():
    # Display progress
    progress_bar = st.progress(0, text="Loading data...")
    
    try:
        # Load the dataset
        df = pd.read_csv('metadata.csv', low_memory=False)
        progress_bar.progress(30, text="Data loaded. Processing...")
        
        # Basic info
        rows, cols = df.shape
        progress_bar.progress(60, text=f"Dataset with {rows} rows and {cols} columns loaded.")
        
        # Check for important columns
        important_cols = ['title', 'abstract', 'publish_time', 'journal', 'authors']
        missing_cols = [col for col in important_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing important columns: {missing_cols}")
            return None
        
        progress_bar.progress(100, text="Data processing complete!")
        return df
    
    except FileNotFoundError:
        st.error("File 'metadata.csv' not found. Please make sure it's in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", 
        "Data Cleaning", 
        "Analysis & Visualizations", 
        "Interactive Explorer",
        "Documentation"
    ])

    with tab1:
        st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Dimensions")
            st.write(f"Number of rows: {df.shape[0]:,}")
            st.write(f"Number of columns: {df.shape[1]}")
            
            st.subheader("Data Types")
            dtypes = df.dtypes.value_counts()
            for dtype, count in dtypes.items():
                st.write(f"{dtype}: {count}")
        
        with col2:
            st.subheader("Missing Values in Key Columns")
            important_cols = ['title', 'abstract', 'publish_time', 'journal', 'authors']
            missing_data = {}
            
            for col in important_cols:
                missing_percent = (df[col].isnull().sum() / len(df)) * 100
                missing_data[col] = missing_percent
            
            missing_df = pd.DataFrame.from_dict(missing_data, orient='index', columns=['Missing (%)'])
            st.dataframe(missing_df.style.format({'Missing (%)': '{:.2f}%'}).highlight_max(color='#ffcccc'))
        
        st.subheader("First 10 Rows of Data")
        st.dataframe(df.head(10))

    with tab2:
        st.markdown('<h2 class="section-header">Data Cleaning</h2>', unsafe_allow_html=True)
        
        # Create a copy for cleaning
        df_clean = df.copy()
        
        # Handle missing values
        st.subheader("Handling Missing Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to drop rows with missing titles
            if st.checkbox("Drop rows with missing titles", value=True):
                initial_count = len(df_clean)
                df_clean = df_clean.dropna(subset=['title'])
                st.write(f"Removed {initial_count - len(df_clean)} rows with missing titles")
        
        with col2:
            # Option to fill missing abstracts
            if st.checkbox("Fill missing abstracts with placeholder", value=True):
                df_clean['abstract'] = df_clean['abstract'].fillna("No abstract available")
        
        # Convert publish_time to datetime and extract year
        st.subheader("Date Conversion")
        
        # Handle date conversion with error handling
        def convert_date(date_val):
            try:
                if isinstance(date_val, str):
                    # Try to parse the date
                    if len(date_val) == 4:  # Only year
                        return datetime.strptime(date_val, '%Y')
                    else:
                        # Try different formats
                        for fmt in ('%Y-%m-%d', '%Y-%m', '%Y/%m/%d', '%Y/%m', '%d-%m-%Y', '%m-%d-%Y'):
                            try:
                                return datetime.strptime(date_val, fmt)
                            except ValueError:
                                continue
                return pd.NaT
            except:
                return pd.NaT
        
        df_clean['publish_time_parsed'] = df_clean['publish_time'].apply(convert_date)
        df_clean['year'] = df_clean['publish_time_parsed'].dt.year
        
        # Count successful conversions
        success_count = df_clean['year'].notna().sum()
        st.write(f"Successfully converted {success_count} of {len(df_clean)} dates ({success_count/len(df_clean)*100:.1f}%)")
        
        # Create abstract word count
        df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
        
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean[['title', 'journal', 'year', 'abstract_word_count']].head(10))

    with tab3:
        st.markdown('<h2 class="section-header">Analysis & Visualizations</h2>', unsafe_allow_html=True)
        
        # Filter out rows without year
        df_with_year = df_clean[df_clean['year'].notna()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Publications by Year")
            year_counts = df_with_year['year'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(year_counts.index.astype(int), year_counts.values, color='steelblue')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Publications')
            ax.set_title('COVID-19 Publications Over Time')
            ax.tick_params(axis='x', rotation=45)
            
            # Set x-axis to integer values
            ax.set_xticks(year_counts.index.astype(int))
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top Journals")
            journal_counts = df_clean['journal'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(journal_counts)), journal_counts.values, color='orange')
            ax.set_yticks(range(len(journal_counts)))
            ax.set_yticklabels(journal_counts.index)
            ax.set_xlabel('Number of Publications')
            ax.set_title('Top 10 Journals by Publication Count')
            
            st.pyplot(fig)
        
        # Word cloud of titles
        st.subheader("Word Cloud of Paper Titles")
        
        # Combine all titles
        all_titles = ' '.join(df_clean['title'].dropna().astype(str))
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Frequent Words in Paper Titles')
        
        st.pyplot(fig)
        
        # Distribution of abstract word count
        st.subheader("Distribution of Abstract Word Count")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_clean['abstract_word_count'], bins=50, color='green', alpha=0.7)
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Abstract Length')
        
        # Remove outliers for better visualization
        ax.set_xlim(0, 500)
        
        st.pyplot(fig)

    with tab4:
        st.markdown('<h2 class="section-header">Interactive Explorer</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Filters")
            
            # Year range slider
            min_year = int(df_clean['year'].min()) if not df_clean['year'].isna().all() else 2019
            max_year = int(df_clean['year'].max()) if not df_clean['year'].isna().all() else 2022
            
            year_range = st.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Journal selector
            top_journals = df_clean['journal'].value_counts().head(20).index.tolist()
            selected_journals = st.multiselect(
                "Select Journals",
                options=top_journals,
                default=top_journals[:3]
            )
            
            # Abstract word count filter
            min_words, max_words = st.slider(
                "Abstract Word Count Range",
                min_value=0,
                max_value=500,
                value=(0, 200)
            )
        
        with col2:
            # Apply filters
            filtered_df = df_clean.copy()
            
            # Filter by year
            filtered_df = filtered_df[
                (filtered_df['year'] >= year_range[0]) & 
                (filtered_df['year'] <= year_range[1])
            ]
            
            # Filter by journal
            if selected_journals:
                filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
            
            # Filter by abstract word count
            filtered_df = filtered_df[
                (filtered_df['abstract_word_count'] >= min_words) & 
                (filtered_df['abstract_word_count'] <= max_words)
            ]
            
            st.subheader(f"Filtered Data: {len(filtered_df)} Papers")
            
            # Show sample of filtered data
            st.dataframe(
                filtered_df[['title', 'journal', 'year', 'abstract_word_count']].head(10),
                height=300
            )
            
            # Show some statistics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Papers", len(filtered_df))
            
            with col_b:
                avg_words = filtered_df['abstract_word_count'].mean()
                st.metric("Avg. Abstract Words", f"{avg_words:.1f}")
            
            with col_c:
                if len(filtered_df) > 0:
                    recent_year = filtered_df['year'].max()
                    st.metric("Most Recent Year", int(recent_year))
                else:
                    st.metric("Most Recent Year", "N/A")

    with tab5:
        st.markdown('<h2 class="section-header">Documentation</h2>', unsafe_allow_html=True)
        
        st.subheader("Project Overview")
        st.markdown("""
        This application analyzes the CORD-19 dataset metadata, which contains information about COVID-19 research papers.
        
        Key Features:
        - Data loading and basic exploration
        - Data cleaning and preprocessing
        - Visualization of publication trends
        - Interactive filtering of papers
        - Word frequency analysis
        """)
        
        st.subheader Methodology")
        st.markdown("""
        1. Data Loading: The metadata.csv file is loaded using pandas with appropriate error handling
        2. Data Cleaning: 
           - Missing values are handled based on user selection
           - Dates are parsed and years are extracted
           - Abstract word counts are calculated
        3. Analysis:
           - Temporal trends in publications
           - Journal distribution
           - Word frequency in titles
        4. Visualization: Using Matplotlib and WordCloud for clear data representation
        """)
        
        st.subheader("Challenges & Solutions")
        st.markdown("""
        Challenge: Inconsistent date formats in the publish_time column
        Solution: Implemented multiple parsing strategies with error handling
        
        Challenge: Large dataset size affecting performance
        Solution: Used Streamlit caching and efficient filtering
        
        Challenge: Missing data in important columns
        Solution: Provided options to handle missing values based on analysis needs
        """)
        
        st.subheader("Future Enhancements")
        st.markdown("""
        - Add more advanced text analysis (NLP techniques)
        - Include citation analysis if data is available
        - Implement topic modeling for abstract text
        - Add network visualization of author collaborations
        - Enable PDF content analysis for available papers
        """)

else:
    st.error("Unable to load the dataset. Please check that 'metadata.csv' is available.")

# Add footer
st.markdown("---")
st.markdown("### CORD-19 Data Explorer | Created with Streamlit")