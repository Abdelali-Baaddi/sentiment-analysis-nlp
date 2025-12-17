"""
Streamlit Web Application for Sentiment Analysis
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import SentimentPredictor
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the sentiment predictor"""
    try:
        predictor = SentimentPredictor(
            model_path='../models/best_model.pkl',
            vectorizer_path='../models/tfidf_vectorizer.pkl'
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">😊 Sentiment Analysis App 😢</h1>', 
                unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Sidebar
    st.sidebar.title("📋 Options")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Text Analysis", "Batch Analysis", "About"]
    )
    
    # Single text analysis
    if app_mode == "Single Text Analysis":
        st.header("📝 Analyze Single Text")
        
        # Text input
        user_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Type or paste your review, comment, or any text here..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and user_input:
            with st.spinner("Analyzing sentiment..."):
                result = predictor.predict(user_input)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment")
                sentiment = result['sentiment']
                if sentiment == 'positive':
                    st.markdown(
                        f'<p class="sentiment-positive">😊 POSITIVE</p>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<p class="sentiment-negative">😢 NEGATIVE</p>', 
                        unsafe_allow_html=True
                    )
            
            with col2:
                if result['confidence']:
                    st.subheader("Confidence")
                    st.markdown(
                        f'<p class="confidence-score">{result["confidence"]:.2%}</p>', 
                        unsafe_allow_html=True
                    )
                    
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'] * 100,
                        title={'text': "Confidence Level"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "lightblue"},
                                {'range': [75, 100], 'color': "royalblue"}
                            ],
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Batch analysis
    elif app_mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        st.write("Upload a CSV file with a 'text' column or enter multiple texts.")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column!")
                else:
                    st.success(f"✅ Loaded {len(df)} texts")
                    
                    if st.button("🔍 Analyze All"):
                        with st.spinner("Analyzing all texts..."):
                            results = predictor.predict_batch(df['text'].tolist())
                        
                        # Add results to dataframe
                        df['sentiment'] = [r['sentiment'] for r in results]
                        df['confidence'] = [r['confidence'] for r in results]
                        
                        # Display results
                        st.subheader("Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Texts", len(df))
                        
                        with col2:
                            positive_count = (df['sentiment'] == 'positive').sum()
                            st.metric("Positive", positive_count)
                        
                        with col3:
                            negative_count = (df['sentiment'] == 'negative').sum()
                            st.metric("Negative", negative_count)
                        
                        # Visualization
                        sentiment_counts = df['sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="sentiment_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Manual batch input
        st.subheader("Or Enter Multiple Texts")
        batch_input = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="Enter one text per line..."
        )
        
        if st.button("🔍 Analyze Batch") and batch_input:
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results = predictor.predict_batch(texts)
            
            # Create dataframe
            results_df = pd.DataFrame(results)
            
            st.subheader("Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            col1, col2 = st.columns(2)
            with col1:
                positive = sum(1 for r in results if r['sentiment'] == 'positive')
                st.metric("Positive", positive)
            with col2:
                negative = sum(1 for r in results if r['sentiment'] == 'negative')
                st.metric("Negative", negative)
    
    # About
    elif app_mode == "About":
        st.header("ℹ️ About")
        st.write("""
        ### Sentiment Analysis App
        
        This application performs sentiment analysis on text data using machine learning.
        
        **Features:**
        - ✅ Single text analysis
        - ✅ Batch processing
        - ✅ Confidence scores
        - ✅ Visual analytics
        - ✅ CSV export
        
        **Model Information:**
        - Trained on movie reviews dataset
        - Uses TF-IDF vectorization
        - Logistic Regression classifier
        
        **How to Use:**
        1. Choose analysis mode from sidebar
        2. Enter text or upload CSV file
        3. Click analyze button
        4. View results and download if needed
        
        ---
        Made with ❤️ using Streamlit
        """)


if __name__ == "__main__":
    main()