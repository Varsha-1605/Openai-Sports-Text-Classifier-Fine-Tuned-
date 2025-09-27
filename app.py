import streamlit as st
import openai
from openai import OpenAI
import time
import pandas as pd
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Sports Text Classifier",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .baseball-prediction {
        background-color: #e8f4fd;
        border: 2px solid #1f77b4;
        color: #1f77b4;
    }
    .hockey-prediction {
        background-color: #fff2e8;
        border: 2px solid #ff7f0e;
        color: #ff7f0e;
    }
    .uncertain-prediction {
        background-color: #f0f0f0;
        border: 2px solid #666666;
        color: #666666;
    }
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

class SportsClassifier:
    def __init__(self, api_key, model_id):
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.valid_labels = ['baseball', 'hockey']
    
    def predict(self, text, temperature=0, max_tokens=5):
        """Make prediction with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            prediction = response.choices[0].message.content.strip().lower()
            
            # Post-processing: handle out-of-domain predictions
            if prediction not in self.valid_labels:
                prediction = "uncertain"
            
            return {
                'prediction': prediction,
                'confidence': 'high' if prediction in self.valid_labels else 'low',
                'raw_response': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'
            }
            
        except Exception as e:
            return {
                'prediction': 'error',
                'confidence': 'none',
                'raw_response': str(e),
                'tokens_used': 0
            }
    
    def batch_predict(self, texts, temperature=0):
        """Predict multiple texts with progress tracking"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, text in enumerate(texts):
            status_text.text(f'Processing text {i+1} of {len(texts)}...')
            result = self.predict(text, temperature)
            result['text'] = text[:50] + "..." if len(text) > 50 else text
            result['timestamp'] = datetime.now()
            results.append(result)
            progress_bar.progress((i + 1) / len(texts))
            time.sleep(0.1)  # Small delay to show progress
        
        progress_bar.empty()
        status_text.empty()
        return results

def main():
    # Header
    st.markdown('<h1 class="main-header">⚾🏒 Sports Text Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the classifier"
        )
        
        # Model ID input
        model_id = st.text_input(
            "Fine-tuned Model ID",
            value="ft:gpt-3.5-turbo-0125:personal::CK2Uuq9e",
            help="Your fine-tuned model ID from OpenAI"
        )
        
        if api_key and model_id:
            if st.button("🔗 Connect to Model"):
                try:
                    st.session_state.client = SportsClassifier(api_key, model_id)
                    st.session_state.api_key_set = True
                    st.success("✅ Connected successfully!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        # Model info
        if st.session_state.api_key_set:
            st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
            st.markdown("**Model Status**: 🟢 Connected")
            st.markdown(f"**Model**: {model_id.split(':')[-1]}")
            st.markdown("**Classes**: Baseball, Hockey")
            st.markdown("**Max Tokens**: 5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Higher values make output more random"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Main content
    if not st.session_state.api_key_set:
        st.warning("⚠️ Please enter your OpenAI API key and connect to the model in the sidebar.")
        
        # Demo section without API
        st.subheader("🎯 About This Classifier")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚾ Baseball Examples:**")
            st.code("The pitcher threw a fastball down the middle.")
            st.code("She hit a home run over the fence.")
            st.code("The batter struck out looking.")
        
        with col2:
            st.markdown("**🏒 Hockey Examples:**")
            st.code("The goalie made an incredible save.")
            st.code("The defenseman cleared the puck.")
            st.code("He scored on a power play.")
        
        return
    
    # Tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📝 Batch Processing", "📊 Analytics", "🧪 Test Cases"])
    
    with tab1:
        st.subheader("Single Text Classification")
        
        # Text input
        text_input = st.text_area(
            "Enter sports-related text:",
            height=150,
            placeholder="e.g., The goalie blocked the shot with his stick..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Classify", type="primary"):
                if text_input.strip():
                    with st.spinner("Classifying..."):
                        result = st.session_state.client.predict(text_input, temperature)
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'text': text_input,
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'timestamp': datetime.now(),
                            'tokens_used': result['tokens_used']
                        })
                        
                        # Display result
                        pred = result['prediction']
                        if pred == 'baseball':
                            st.markdown(f'<div class="prediction-box baseball-prediction">⚾ BASEBALL</div>', 
                                      unsafe_allow_html=True)
                        elif pred == 'hockey':
                            st.markdown(f'<div class="prediction-box hockey-prediction">🏒 HOCKEY</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="prediction-box uncertain-prediction">❓ {pred.upper()}</div>', 
                                      unsafe_allow_html=True)
                        
                        # Additional info
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence", result['confidence'])
                        with col_b:
                            st.metric("Tokens Used", result['tokens_used'])
                else:
                    st.warning("Please enter some text to classify.")
        
        with col2:
            if st.button("🎲 Random Example"):
                examples = [
                    "The pitcher threw a curveball for strike three.",
                    "The goalie sprawled to make the save.",
                    "The runner stole second base successfully.",
                    "The power play resulted in a goal.",
                    "The batter hit a double down the line.",
                    "The defenseman checked the forward into the boards.",
                    "The closer came in to finish the ninth inning.",
                    "The penalty shot was saved by the keeper."
                ]
                import random
                st.text_area("Random example:", value=random.choice(examples), key="random_example")
    
    with tab2:
        st.subheader("Batch Text Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a text file or CSV",
            type=['txt', 'csv'],
            help="For CSV files, make sure there's a 'text' column"
        )
        
        # Manual text input
        batch_text = st.text_area(
            "Or enter multiple texts (one per line):",
            height=200,
            placeholder="The pitcher threw a fastball...\nThe goalie made a save...\nThe batter struck out..."
        )
        
        if st.button("🔄 Process Batch"):
            texts = []
            
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts = df['text'].dropna().tolist()
                    else:
                        st.error("CSV file must have a 'text' column")
                        return
                else:
                    texts = uploaded_file.read().decode('utf-8').split('\n')
            elif batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            if texts:
                results = st.session_state.client.batch_predict(texts, temperature)
                
                # Display results
                df_results = pd.DataFrame(results)
                st.dataframe(df_results[['text', 'prediction', 'confidence', 'tokens_used']], 
                           use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    baseball_count = sum(1 for r in results if r['prediction'] == 'baseball')
                    st.metric("Baseball", baseball_count)
                with col3:
                    hockey_count = sum(1 for r in results if r['prediction'] == 'hockey')
                    st.metric("Hockey", hockey_count)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "classification_results.csv",
                    "text/csv"
                )
    
    with tab3:
        st.subheader("Prediction Analytics")
        
        if st.session_state.prediction_history:
            df_history = pd.DataFrame(st.session_state.prediction_history)
            
            # Prediction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    df_history, 
                    names='prediction', 
                    title='Prediction Distribution',
                    color_discrete_map={
                        'baseball': '#1f77b4',
                        'hockey': '#ff7f0e',
                        'uncertain': '#666666',
                        'error': '#d62728'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Timeline
                df_history['hour'] = pd.to_datetime(df_history['timestamp']).dt.floor('H')
                timeline_data = df_history.groupby(['hour', 'prediction']).size().reset_index(name='count')
                
                fig_timeline = px.bar(
                    timeline_data,
                    x='hour',
                    y='count',
                    color='prediction',
                    title='Predictions Over Time',
                    color_discrete_map={
                        'baseball': '#1f77b4',
                        'hockey': '#ff7f0e',
                        'uncertain': '#666666'
                    }
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Recent predictions table
            st.subheader("Recent Predictions")
            recent_df = df_history.tail(10)[['text', 'prediction', 'confidence', 'timestamp']]
            recent_df['text'] = recent_df['text'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
            st.dataframe(recent_df, use_container_width=True)
            
            # Token usage
            if 'tokens_used' in df_history.columns:
                total_tokens = df_history['tokens_used'].sum()
                avg_tokens = df_history['tokens_used'].mean()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tokens Used", f"{total_tokens:,}")
                with col2:
                    st.metric("Average Tokens per Request", f"{avg_tokens:.1f}")
            
        else:
            st.info("No predictions made yet. Use the Single Prediction or Batch Processing tabs to get started!")
    
    with tab4:
        st.subheader("Edge Case Testing")
        st.info("Test the model on ambiguous or challenging examples")
        
        # Predefined test cases
        test_cases = {
            "Clear Baseball": [
                "The pitcher threw a fastball down the middle.",
                "She hit a home run over the fence.",
                "The batter struck out swinging on three pitches."
            ],
            "Clear Hockey": [
                "The goalie made an incredible save.",
                "The defenseman cleared the puck from the zone.",
                "He scored on the power play with a slap shot."
            ],
            "Ambiguous/Edge Cases": [
                "The player scored in the final seconds.",
                "The team celebrated their championship victory.",
                "He was injured during the match.",
                "The coach gave a motivational speech.",
                "The referee made a controversial call.",
                "The match ended in overtime.",
                "They played outdoors on a sunny day."
            ]
        }
        
        for category, cases in test_cases.items():
            st.subheader(f"📋 {category}")
            
            if st.button(f"Test All {category}", key=f"test_{category}"):
                results = st.session_state.client.batch_predict(cases, temperature)
                
                for case, result in zip(cases, results):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"*{case}*")
                    with col2:
                        pred = result['prediction']
                        if pred == 'baseball':
                            st.markdown("⚾ **Baseball**")
                        elif pred == 'hockey':
                            st.markdown("🏒 **Hockey**")
                        else:
                            st.markdown(f"❓ **{pred.title()}**")
                    st.divider()

if __name__ == "__main__":
    main()