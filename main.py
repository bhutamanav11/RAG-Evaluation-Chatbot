from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
import tempfile
import shutil
import uuid
import numpy as np
import io
import csv
import json

# Import our models
from models.database import DatabaseManager
from models.vector_store import VectorStore
from models.document_processor import DocumentProcessor
from models.llm_providers import LLMProvider
from models.evaluator import RAGEvaluator
from config import Config

app = FastAPI(title="RAG Evaluation System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config_dict = {
    'OPENAI_API_KEY': Config.OPENAI_API_KEY,
    'GOOGLE_API_KEY': Config.GOOGLE_API_KEY,
    'ANTHROPIC_API_KEY': Config.ANTHROPIC_API_KEY
}

db_manager = DatabaseManager()
vector_store = VectorStore(Config.CHROMA_PERSIST_DIR, Config.EMBEDDING_MODEL)
document_processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
llm_provider = LLMProvider(config_dict)
evaluator = RAGEvaluator(Config.EMBEDDING_MODEL)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_name: str = "gemini-pro"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_number: int
    model_name: str
    retrieved_chunks: List[Dict[str, Any]]
    evaluation_scores: Dict[str, Any]
    response_time: float

class EvaluationRequest(BaseModel):
    session_id: str
    model_names: List[str] = ["gemini-pro", "claude-3-haiku-20240307","gpt-3.5-turbo"]

class RemoveDocumentRequest(BaseModel):
    filename: str

@app.post("/api/v1/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        processed_docs = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                # Process document
                doc_data = document_processor.process_document(tmp_path, file.filename)
                
                # Add to vector store
                vector_ids = vector_store.add_documents(doc_data['chunks'])
                
                # Save to database
                document_id = db_manager.add_document_record(
                    file.filename, 
                    doc_data['file_type'], 
                    doc_data['chunk_count']
                )
                
                processed_docs.append({
                    'document_id': document_id,
                    'filename': file.filename,
                    'chunk_count': doc_data['chunk_count'],
                    'file_type': doc_data['file_type']
                })
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        
        return {
            'success': True,
            'processed_documents': processed_docs,
            'total_chunks': sum(doc['chunk_count'] for doc in processed_docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat conversation"""
    try:
        # Get conversation history
        history = db_manager.get_session_history(request.session_id)
        turn_number = len(history) + 1
        
        # Retrieve relevant chunks
        retrieved_chunks = vector_store.similarity_search(request.message, k=Config.TOP_K_RETRIEVAL)
        chunk_texts = [chunk['content'] for chunk in retrieved_chunks]
        
        # Format conversation history for LLM
        messages = []
        for turn in history[-5:]:  # Last 5 turns for context
            messages.append({'role': 'user', 'content': turn['user_message']})
            messages.append({'role': 'assistant', 'content': turn['assistant_response']})
        messages.append({'role': 'user', 'content': request.message})
        
        # Generate response
        response_data = await llm_provider.generate_response(
            request.model_name, messages, chunk_texts
        )
        
        if not response_data['success']:
            raise HTTPException(status_code=500, detail=response_data.get('error', 'Unknown error'))
        
        # Evaluate response
        evaluation_scores = await evaluator.evaluate_response(
            request.message, response_data['response'], chunk_texts, history, request.model_name
        )
        
        # Save conversation to database
        conversation_id = db_manager.add_conversation(
            request.session_id, turn_number, request.message,
            response_data['response'], request.model_name, retrieved_chunks,
            response_data['response_time'], response_data['token_count']
        )
        
        # Save evaluation
        db_manager.add_evaluation(conversation_id, request.model_name, evaluation_scores)
        
        return ChatResponse(
            response=response_data['response'],
            session_id=request.session_id,
            turn_number=turn_number,
            model_name=request.model_name,
            retrieved_chunks=retrieved_chunks,
            evaluation_scores=evaluation_scores,
            response_time=response_data['response_time']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sessions")
async def create_session():
    """Create new conversation session"""
    session_id = db_manager.create_session()
    return {'session_id': session_id}

@app.get("/api/v1/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    history = db_manager.get_session_history(session_id)
    return {'session_id': session_id, 'history': history}

@app.post("/api/v1/evaluate")
async def run_comparative_evaluation(request: EvaluationRequest):
    """Run comparative evaluation across multiple models"""
    try:
        # Get session history
        history = db_manager.get_session_history(request.session_id)
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        results = {}
        
        # Evaluate each model on the same conversation
        for model_name in request.model_names:
            model_results = []
            
            for turn in history:
                # Re-evaluate this turn with the specified model
                retrieved_chunks = eval(turn['retrieved_chunks']) if turn['retrieved_chunks'] else []
                chunk_texts = [chunk['content'] for chunk in retrieved_chunks]
                
                # Get conversation context up to this turn
                context_history = [h for h in history if h['turn_number'] < turn['turn_number']]
                
                evaluation_scores = await evaluator.evaluate_response(
                    turn['user_message'], turn['assistant_response'], 
                    chunk_texts, context_history, model_name
                )
                
                model_results.append({
                    'turn_number': turn['turn_number'],
                    'scores': evaluation_scores
                })
            
            # Calculate degradation curve
            degradation_analysis = evaluator.calculate_degradation_curve(model_results)
            
            results[model_name] = {
                'turn_results': model_results,
                'degradation_analysis': degradation_analysis,
                'average_scores': {
                    metric: float(np.mean([turn['scores'][metric] for turn in model_results 
                                   if isinstance(turn['scores'][metric], (int, float))]))
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 
                                  'context_recall', 'context_degradation', 'chunk_efficiency']
                }
            }
        
        return {
            'session_id': request.session_id,
            'evaluation_results': results,
            'total_turns': len(history)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/export/{session_id}/csv")
async def export_session_csv(session_id: str):
    """Export session data to CSV"""
    try:
        # Get conversation history
        history = db_manager.get_session_history(session_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow([
            'Turn Number', 'Model', 'User Question', 'Assistant Response',
            'Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall',
            'Context Degradation', 'Chunk Efficiency', 'Failure Mode',
            'Response Time (s)', 'Token Count', 'Timestamp'
        ])
        
        # Write data
        for turn in history:
            # Get evaluation scores
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM evaluations 
                    WHERE conversation_id = ?
                """, (turn['conversation_id'],))
                eval_row = cursor.fetchone()
            
            if eval_row:
                eval_data = dict(eval_row)
                writer.writerow([
                    turn['turn_number'],
                    turn['model_used'],
                    turn['user_message'],
                    turn['assistant_response'],
                    eval_data.get('faithfulness_score', 0),
                    eval_data.get('answer_relevancy', 0),
                    eval_data.get('context_precision', 0),
                    eval_data.get('context_recall', 0),
                    eval_data.get('context_degradation_score', 0),
                    eval_data.get('chunk_efficiency', 0),
                    eval_data.get('failure_mode', 'none'),
                    turn['response_time'],
                    turn['token_count'],
                    turn['created_at']
                ])
        
        # Prepare response
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=session_{session_id[:8]}_data.csv"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/export/{session_id}/json")
async def export_session_json(session_id: str):
    """Export session data to JSON"""
    try:
        # Get conversation history
        history = db_manager.get_session_history(session_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Build complete session data
        session_data = {
            'session_id': session_id,
            'total_turns': len(history),
            'conversations': []
        }
        
        for turn in history:
            # Get evaluation scores
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM evaluations 
                    WHERE conversation_id = ?
                """, (turn['conversation_id'],))
                eval_row = cursor.fetchone()
            
            turn_data = {
                'turn_number': turn['turn_number'],
                'model': turn['model_used'],
                'user_message': turn['user_message'],
                'assistant_response': turn['assistant_response'],
                'response_time': turn['response_time'],
                'token_count': turn['token_count'],
                'timestamp': turn['created_at'],
                'evaluation': {
                    'faithfulness': dict(eval_row).get('faithfulness_score', 0) if eval_row else 0,
                    'answer_relevancy': dict(eval_row).get('answer_relevancy', 0) if eval_row else 0,
                    'context_precision': dict(eval_row).get('context_precision', 0) if eval_row else 0,
                    'context_recall': dict(eval_row).get('context_recall', 0) if eval_row else 0,
                    'context_degradation': dict(eval_row).get('context_degradation_score', 0) if eval_row else 0,
                    'chunk_efficiency': dict(eval_row).get('chunk_efficiency', 0) if eval_row else 0,
                    'failure_mode': dict(eval_row).get('failure_mode', 'none') if eval_row else 'none'
                } if eval_row else {}
            }
            
            session_data['conversations'].append(turn_data)
        
        return JSONResponse(
            content=session_data,
            headers={
                "Content-Disposition": f"attachment; filename=session_{session_id[:8]}_data.json"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/degradation-curves")
async def get_degradation_analytics():
    """Get degradation analysis across all sessions"""
    try:
        # Get all evaluations
        evaluations = db_manager.get_evaluation_results()
        
        # Group by model and session
        model_analytics = {}
        
        for eval_data in evaluations:
            model = eval_data['model_name']
            if model not in model_analytics:
                model_analytics[model] = {
                    'total_conversations': 0,
                    'average_degradation_rate': 0,
                    'failure_modes': {},
                    'performance_metrics': {}
                }
            
            # Count failure modes
            failure_mode = eval_data.get('failure_mode', 'none')
            model_analytics[model]['failure_modes'][failure_mode] = \
                model_analytics[model]['failure_modes'].get(failure_mode, 0) + 1
        
        return {'model_analytics': model_analytics}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS (NEW)
# ============================================================================

@app.get("/api/v1/database/list-documents")
async def list_documents():
    """List all documents with metadata"""
    try:
        # Get from database
        documents = db_manager.get_all_documents()
        
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats()
        
        return {
            'success': True,
            'documents': documents,
            'total_chunks': vector_stats['total_documents']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/remove-document")
async def remove_document(request: RemoveDocumentRequest):
    """Remove a specific document from vector store and database"""
    try:
        filename = request.filename
        
        # Get all chunks with this filename from vector store
        all_data = vector_store.collection.get(include=['metadatas'])
        
        # Find IDs to delete
        ids_to_delete = [
            all_data['ids'][i]
            for i in range(len(all_data['ids']))
            if all_data['metadatas'][i].get('filename') == filename
        ]
        
        # Delete from vector store
        if ids_to_delete:
            vector_store.collection.delete(ids=ids_to_delete)
        
        # Remove from database
        db_manager.remove_document_record(filename)
        
        return {
            'success': True,
            'message': f'Removed {len(ids_to_delete)} chunks from {filename}',
            'removed_chunks': len(ids_to_delete)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/clear-all")
async def clear_all_documents():
    """Clear ALL documents from vector store and database"""
    try:
        # Get count before clearing
        count_before = vector_store.collection.count()
        
        # Delete collection and recreate
        vector_store.client.delete_collection("documents")
        vector_store.collection = vector_store.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear database records
        db_manager.clear_all_documents()
        
        return {
            'success': True,
            'message': f'Cleared {count_before} chunks from all documents',
            'removed_chunks': count_before
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/database/reset-evaluations")
async def reset_evaluations():
    """Reset ALL evaluation records - preserves documents"""
    try:
        db_manager.reset_all_evaluations()
        
        return {
            'success': True,
            'message': 'All evaluation records cleared. Documents preserved.'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/database/stats")
async def get_database_stats():
    """Get comprehensive database statistics"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count evaluations
            cursor.execute("SELECT COUNT(*) as count FROM evaluations")
            eval_count = cursor.fetchone()[0]
            
            # Count conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conv_count = cursor.fetchone()[0]
            
            # Count sessions
            cursor.execute("SELECT COUNT(*) as count FROM sessions")
            session_count = cursor.fetchone()[0]
            
            # Count documents
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Vector store stats
            vector_stats = vector_store.get_collection_stats()
        
        return {
            'success': True,
            'evaluations': eval_count,
            'conversations': conv_count,
            'sessions': session_count,
            'documents': doc_count,
            'vector_chunks': vector_stats['total_documents']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    vector_stats = vector_store.get_collection_stats()
    
    # Get available models based on what's configured
    available_models = []
    for model_name in Config.get_enabled_models():
        if llm_provider.is_model_available(model_name):
            available_models.append(model_name)
    
    return {
        'status': 'healthy',
        'vector_store': vector_stats,
        'available_models': available_models,
        'configured_providers': llm_provider.available_providers
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)