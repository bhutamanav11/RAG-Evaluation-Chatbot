import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import time
import os
import csv
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page configuration
st.set_page_config(
    page_title="AI Models Evaluation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    max-width: 1400px;
    margin: 0 auto;
}

.chat-message {
    padding: 1.2rem;
    margin: 0.8rem 0;
    border-radius: 8px;
    border-left: 3px solid;
}

.user-message {
    border-left-color: #1976D2;
    background-color: rgba(25, 118, 210, 0.05);
}

.assistant-message {
    border-left-color: #7B1FA2;
    background-color: rgba(123, 31, 162, 0.05);
}

.message-label {
    font-weight: 600;
    font-size: 0.8rem;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.message-content {
    color: #ffffff;
    line-height: 1.6;
    font-size: 0.95rem;
}

.evaluation-scores {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.78rem;
    color: #b0b0b0;
    line-height: 1.4;
}

.score-inline {
    display: inline;
    margin-right: 1.2rem;
    white-space: nowrap;
}

.score-label {
    font-weight: 500;
    color: #999;
}

.score-value {
    font-weight: 600;
    color: #e0e0e0;
    margin-left: 0.3rem;
}

.metric-card {
    background-color: rgba(0, 0, 0, 0.02);
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
}

h1, h2, h3 {
    font-weight: 600;
    color: #1a1a1a;
}

h2 {
    font-size: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        response = requests.post(f"{API_BASE_URL}/sessions")
        st.session_state.session_id = response.json()['session_id']
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_uploaded' not in st.session_state:
        st.session_state.documents_uploaded = False
    
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []

def save_conversation_to_csv(session_id: str, chat_history: list):
    """Auto-save ALL conversations to a single master CSV file"""
    os.makedirs("exports", exist_ok=True)
    master_file = "exports/all_conversations.csv"
    file_exists = os.path.isfile(master_file)
    
    with open(master_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'Session ID', 'Timestamp', 'Turn Number', 'Model',
                'User Question', 'Assistant Response',
                'Faithfulness', 'Answer Relevancy', 'Context Degradation',
                'Chunk Efficiency', 'Failure Mode', 'Response Time (s)'
            ])
        
        if chat_history:
            message = chat_history[-1]
            scores = message.get('evaluation_scores', {})
            
            writer.writerow([
                session_id[:8],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                message.get('turn_number', len(chat_history)),
                message['model_name'],
                message['user_message'],
                message['assistant_response'],
                f"{scores.get('faithfulness', 0):.3f}",
                f"{scores.get('answer_relevancy', 0):.3f}",
                f"{scores.get('context_degradation', 0):.3f}",
                f"{scores.get('chunk_efficiency', 0):.3f}",
                scores.get('failure_mode', 'none'),
                f"{message.get('response_time', 0):.2f}"
            ])
    
    return master_file

def upload_documents():
    """Handle document upload"""
    st.subheader("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload documents to use as knowledge base"
    )
    
    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            files = []
            for uploaded_file in uploaded_files:
                files.append(('files', (uploaded_file.name, uploaded_file.read(), uploaded_file.type)))
            
            response = requests.post(f"{API_BASE_URL}/upload-documents", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Successfully processed {len(result['processed_documents'])} documents")
                
                st.session_state.documents_uploaded = True
                st.session_state.processed_docs = result['processed_documents']
                
                st.write("**Processed Documents:**")
                df = pd.DataFrame(result['processed_documents'])
                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"❌ Error processing documents: {response.text}")
    
    elif st.session_state.processed_docs:
        st.write("**Currently Loaded Documents:**")
        df = pd.DataFrame(st.session_state.processed_docs)
        st.dataframe(df, use_container_width=True)

def document_management_page():
    """Document Management - View and delete documents"""
    st.subheader("📚 Document Management")
    
    # Get stats
    try:
        stats_response = requests.get(f"{API_BASE_URL}/database/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Documents", stats['documents'])
            with col2:
                st.metric("🔢 Vector Chunks", stats['vector_chunks'])
            with col3:
                st.metric("💬 Conversations", stats['conversations'])
            with col4:
                st.metric("📊 Evaluations", stats['evaluations'])
            
            st.divider()
    except:
        st.error("❌ Could not connect to API")
        return
    
    # Document list
    try:
        response = requests.get(f"{API_BASE_URL}/database/list-documents")
        if response.status_code == 200:
            result = response.json()
            documents = result.get('documents', [])
            
            if documents:
                st.write("### 📄 Uploaded Documents")
                
                for doc in documents:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{doc['filename']}**")
                            st.caption(f"Uploaded: {doc['uploaded_at'][:19]}")
                        
                        with col2:
                            st.write(f"**{doc['chunk_count']}** chunks")
                        
                        with col3:
                            st.write(f"Type: `{doc['file_type']}`")
                        
                        with col4:
                            if st.button("🗑️ Remove", key=f"remove_{doc['filename']}"):
                                with st.spinner(f"Removing {doc['filename']}..."):
                                    remove_response = requests.post(
                                        f"{API_BASE_URL}/database/remove-document",
                                        json={"filename": doc['filename']}
                                    )
                                    if remove_response.status_code == 200:
                                        st.success(f"✅ Removed {doc['filename']}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Error: {remove_response.text}")
                        
                        st.divider()
                
                st.write("---")
                
                # Bulk actions
                st.write("### ⚠️ Bulk Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Clear All Documents**")
                    st.caption("Remove ALL documents from the system")
                    if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                        if 'confirm_clear_all' not in st.session_state:
                            st.session_state.confirm_clear_all = True
                            st.warning("⚠️ Click again to confirm!")
                            st.rerun()
                        else:
                            with st.spinner("Clearing all documents..."):
                                clear_response = requests.post(f"{API_BASE_URL}/database/clear-all")
                                if clear_response.status_code == 200:
                                    st.success("✅ All documents cleared!")
                                    st.session_state.documents_uploaded = False
                                    st.session_state.processed_docs = []
                                    st.session_state.pop('confirm_clear_all', None)
                                    time.sleep(1)
                                    st.rerun()
                
                with col2:
                    st.write("**Reset Evaluations Only**")
                    st.caption("Keep documents but clear evaluation history")
                    if st.button("🔄 Reset Evaluations", type="secondary", use_container_width=True):
                        if 'confirm_reset_eval' not in st.session_state:
                            st.session_state.confirm_reset_eval = True
                            st.info("ℹ️ Click again to confirm!")
                            st.rerun()
                        else:
                            with st.spinner("Resetting evaluations..."):
                                reset_response = requests.post(f"{API_BASE_URL}/database/reset-evaluations")
                                if reset_response.status_code == 200:
                                    st.success("✅ Evaluation data cleared! Documents preserved.")
                                    st.session_state.chat_history = []
                                    # Create new session
                                    session_response = requests.post(f"{API_BASE_URL}/sessions")
                                    st.session_state.session_id = session_response.json()['session_id']
                                    st.session_state.pop('confirm_reset_eval', None)
                                    time.sleep(1)
                                    st.rerun()
            else:
                st.info("📭 No documents uploaded yet")
                st.write("Go to **Document Upload** page to add documents")
    
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
    
    # Export section
    st.write("---")
    st.write("### 📊 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download All Conversations (CSV)", use_container_width=True):
            # Check if CSV exists
            if os.path.exists("exports/all_conversations.csv"):
                with open("exports/all_conversations.csv", "rb") as f:
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=f,
                        file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.warning("No conversation data available yet")
    
    with col2:
        st.info("💡 Use the CSV export to analyze conversations offline or in Excel")

def chat_interface():
    """Main chat interface"""
    st.subheader("💬 Chat Interface")
    
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        available_models = health_response.json().get('available_models', [])
        
        if not available_models:
            st.error("❌ No LLM models available. Please configure API keys in .env file")
            st.info("💡 Free options: Gemini Pro (Google)")
            return
    except:
        available_models = ["gemini-pro", "claude-3-haiku-20240307","gpt-3.5-turbo"]
    
    col1, col2= st.columns([3, 1])
    with col2:
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose an LLM model"
        )
    
    with col1:
        if selected_model:
            if 'gemini' in selected_model:
                st.caption("✅ Gemini Pro: FREE tier available")
            elif 'claude' in selected_model:
                st.caption("💰 Claude Haiku: ~$0.25 per million tokens")
            elif 'gpt' in selected_model:
                st.caption("💳 OpenAI: Requires payment")
    
    # Display chat history
    for message in st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-label">You</div>
            <div class="message-content">{message["user_message"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-label">{message["model_name"]}</div>
            <div class="message-content">{message["assistant_response"]}</div>
        """, unsafe_allow_html=True)
        
        if 'evaluation_scores' in message:
            scores = message['evaluation_scores']
            st.markdown(f"""
            <div class="evaluation-scores">
                <span class="score-inline"><span class="score-label">Faithfulness:</span><span class="score-value">{scores.get('faithfulness', 0):.3f}</span></span>
                <span class="score-inline"><span class="score-label">Answer Relevancy:</span><span class="score-value">{scores.get('answer_relevancy', 0):.3f}</span></span>
                <span class="score-inline"><span class="score-label">Context Degradation:</span><span class="score-value">{scores.get('context_degradation', 0):.3f}</span></span>
                <span class="score-inline"><span class="score-label">Response Time:</span><span class="score-value">{message.get('response_time', 0):.2f}s</span></span>
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    # Input section
    user_input = st.text_input("", key=f"chat_input_{len(st.session_state.chat_history)}", placeholder="Ask a question about your documents...", label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        response = requests.post(f"{API_BASE_URL}/sessions")
        st.session_state.session_id = response.json()['session_id']
        st.rerun()
    
    if send_button and user_input:
        if not st.session_state.documents_uploaded:
            st.warning("⚠️ Please upload documents first")
            return
        
        with st.spinner(f"Getting response from {selected_model}..."):
            chat_request = {
                "message": user_input,
                "session_id": st.session_state.session_id,
                "model_name": selected_model
            }
            
            response = requests.post(f"{API_BASE_URL}/chat", json=chat_request)
            
            if response.status_code == 200:
                result = response.json()
                
                new_message = {
                    'user_message': user_input,
                    'assistant_response': result['response'],
                    'model_name': result['model_name'],
                    'evaluation_scores': result['evaluation_scores'],
                    'response_time': result['response_time'],
                    'turn_number': result['turn_number']
                }
                st.session_state.chat_history.append(new_message)
                
                # AUTO-SAVE to single CSV file
                save_conversation_to_csv(st.session_state.session_id, st.session_state.chat_history)
                
                st.rerun()
            else:
                st.error(f"❌ Error: {response.text}")

def evaluation_dashboard():
    """Evaluation and analytics dashboard"""
    st.subheader("📊 Evaluation Dashboard")
    
    if not st.session_state.chat_history:
        st.info("ℹ️ Start a conversation to see evaluation metrics")
        return
    
    st.divider()
    
    # Get available models
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        available_models = health_response.json().get('available_models', [])
    except:
        available_models = ["gemini-pro", "claude-3-haiku-20240307", "gpt-3.5-turbo"]
    
    # Comparative Evaluation Section
    st.write("### 📊 Comparative Model Evaluation")
    st.write("Select models to compare on the same conversation:")
    
    models_to_evaluate = st.multiselect(
        "Select models:",
        available_models,
        default=[available_models[0]] if available_models else []
    )
    
    if st.button("Run Comparative Evaluation", type="primary", disabled=not models_to_evaluate):
        with st.spinner("Running evaluation..."):
            eval_request = {
                "session_id": st.session_state.session_id,
                "model_names": models_to_evaluate
            }
            
            response = requests.post(f"{API_BASE_URL}/evaluate", json=eval_request)
            
            if response.status_code == 200:
                results = response.json()
                st.session_state.evaluation_results = results
                st.success("✅ Evaluation complete!")
            else:
                st.error(f"❌ Error: {response.text}")
    
    st.divider()
    
    # Current Conversation Metrics
    if st.session_state.chat_history:
        st.subheader("Current Conversation Metrics")
        
        metrics_data = []
        for turn in st.session_state.chat_history:
            if 'evaluation_scores' in turn:
                scores = turn['evaluation_scores']
                metrics_data.append({
                    'Turn': turn.get('turn_number', len(metrics_data) + 1),
                    'Model': turn['model_name'],
                    'Faithfulness': scores.get('faithfulness', 0),
                    'Answer Relevancy': scores.get('answer_relevancy', 0),
                    'Context Degradation': scores.get('context_degradation', 0),
                    'Chunk Efficiency': scores.get('chunk_efficiency', 0),
                    'Response Time': turn.get('response_time', 0)
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Degradation curve
            fig_degradation = px.line(
                df_metrics, 
                x='Turn', 
                y='Context Degradation',
                color='Model',
                title='Context Degradation Over Conversation Turns',
                markers=True
            )
            st.plotly_chart(fig_degradation, use_container_width=True)
            
            # Average metrics table
            numeric_cols = ['Faithfulness', 'Answer Relevancy', 'Context Degradation', 
                          'Chunk Efficiency', 'Response Time']
            avg_metrics = df_metrics.groupby('Model')[numeric_cols].mean().round(3)
            st.subheader("Average Metrics by Model")
            st.dataframe(avg_metrics, use_container_width=True)
            
            # Response time comparison
            fig_response_time = px.bar(
                df_metrics,
                x='Model',
                y='Response Time',
                title='Average Response Time by Model'
            )
            st.plotly_chart(fig_response_time, use_container_width=True)
    
    # Comparative Evaluation Results
    if hasattr(st.session_state, 'evaluation_results'):
        st.subheader("Comparative Evaluation Results")
        results = st.session_state.evaluation_results
        
        comparison_data = []
        for model, data in results['evaluation_results'].items():
            avg_scores = data['average_scores']
            degradation = data.get('degradation_analysis', {})
            
            comparison_data.append({
                'Model': model,
                'Avg Faithfulness': avg_scores.get('faithfulness', 0),
                'Avg Relevancy': avg_scores.get('answer_relevancy', 0),
                'Avg Context Degradation': avg_scores.get('context_degradation', 0),
                'Degradation Rate': degradation.get('degradation_rate', 0),
                'Initial Performance': degradation.get('initial_performance', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Radar chart comparison
        if len(comparison_data) > 1:
            metrics = ['Avg Faithfulness', 'Avg Relevancy', 'Avg Context Degradation']
            
            fig_radar = go.Figure()
            
            for _, row in df_comparison.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[metric] for metric in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Performance Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Comparison table
        st.dataframe(df_comparison, use_container_width=True)

def research_analysis():
    """Research analysis and insights"""
    st.subheader("🔬 Research Analysis")
    
    st.write("**Context Degradation Analysis**")
    
    if hasattr(st.session_state, 'evaluation_results'):
        degradation_insights = []
        for model, data in st.session_state.evaluation_results['evaluation_results'].items():
            degradation = data.get('degradation_analysis', {})
            if 'degradation_rate' in degradation:
                degradation_insights.append({
                    'Model': model,
                    'Degradation Rate': degradation['degradation_rate'],
                    'Initial Performance': degradation.get('initial_performance', 0),
                    'R-squared': degradation.get('r_squared', 0)
                })
        
        if degradation_insights:
            df_degradation = pd.DataFrame(degradation_insights)
            
            fig_degradation_rate = px.bar(
                df_degradation,
                x='Model',
                y='Degradation Rate',
                title='Context Degradation Rate by Model (Lower is Better)',
                color='Degradation Rate',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_degradation_rate, use_container_width=True)
            
            st.dataframe(df_degradation, use_container_width=True)
    
    st.write("**Key Research Insights**")
    
    if st.session_state.chat_history:
        total_turns = len(st.session_state.chat_history)
        
        insights = []
        if total_turns >= 5:
            insights.append("✅ Sufficient conversation length for degradation analysis")
        else:
            insights.append("⚠️ Need more conversation turns for reliable degradation analysis")
        
        all_scores = []
        for turn in st.session_state.chat_history:
            if 'evaluation_scores' in turn:
                scores = turn['evaluation_scores']
                all_scores.append(scores.get('context_degradation', 0))
        
        if all_scores:
            avg_degradation = sum(all_scores) / len(all_scores)
            if avg_degradation > 0.7:
                insights.append("✅ Good context retention observed")
            elif avg_degradation > 0.5:
                insights.append("⚠️ Moderate context degradation detected")
            else:
                insights.append("❌ Significant context degradation observed")
        
        for insight in insights:
            st.write(f"- {insight}")

def main():
    """Main application"""
    st.title("🤖 AI Models Evaluation System")
    st.caption("Advanced Research Platform for Evaluating Conversational RAG Systems")
    
    initialize_session()
    
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "Document Upload", 
                "Document Management",
                "Chat Interface", 
                "Evaluation Dashboard", 
                "Research Analysis"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.write("**Session Info**")
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
        st.caption(f"Turns: {len(st.session_state.chat_history)}")
        st.caption(f"Documents: {'✅ Loaded' if st.session_state.documents_uploaded else '❌ None'}")
        
        # Quick stats
        if st.session_state.documents_uploaded:
            st.divider()
            st.write("**Quick Stats**")
            try:
                stats_response = requests.get(f"{API_BASE_URL}/database/stats")
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    st.caption(f"📄 Docs: {stats['documents']}")
                    st.caption(f"💬 Convos: {stats['conversations']}")
                    st.caption(f"📊 Evals: {stats['evaluations']}")
            except:
                pass
    
    # Route to appropriate page
    if page == "Document Upload":
        upload_documents()
    elif page == "Document Management":
        document_management_page()
    elif page == "Chat Interface":
        chat_interface()
    elif page == "Evaluation Dashboard":
        evaluation_dashboard()
    elif page == "Research Analysis":
        research_analysis()

if __name__ == "__main__":
    main()