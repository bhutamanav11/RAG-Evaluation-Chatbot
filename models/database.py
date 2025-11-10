import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path: str = "rag_evaluation.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content TEXT,
                    chunk_count INTEGER,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    turn_number INTEGER,
                    user_message TEXT,
                    assistant_response TEXT,
                    model_used TEXT,
                    retrieved_chunks TEXT,
                    response_time REAL,
                    token_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    model_name TEXT,
                    faithfulness_score REAL,
                    answer_relevancy REAL,
                    context_precision REAL,
                    context_recall REAL,
                    context_degradation_score REAL,
                    failure_mode TEXT,
                    chunk_efficiency REAL,
                    evaluation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            # Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    performance_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    session_id TEXT,
                    avg_response_time REAL,
                    total_tokens INTEGER,
                    total_cost REAL,
                    conversation_turns INTEGER,
                    success_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_session(self, user_id: str = "default") -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
            conn.commit()
        return session_id
    
    def add_conversation(self, session_id: str, turn_number: int, 
                        user_message: str, assistant_response: str,
                        model_used: str, retrieved_chunks: List[Dict],
                        response_time: float, token_count: int) -> str:
        """Add conversation turn"""
        conversation_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (conversation_id, session_id, turn_number, user_message, 
                 assistant_response, model_used, retrieved_chunks, 
                 response_time, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (conversation_id, session_id, turn_number, user_message,
                  assistant_response, model_used, json.dumps(retrieved_chunks),
                  response_time, token_count))
            conn.commit()
        return conversation_id
    
    def add_evaluation(self, conversation_id: str, model_name: str, scores: Dict[str, float]):
        """Add evaluation scores"""
        evaluation_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations 
                (evaluation_id, conversation_id, model_name, faithfulness_score,
                 answer_relevancy, context_precision, context_recall,
                 context_degradation_score, failure_mode, chunk_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (evaluation_id, conversation_id, model_name, 
                  scores.get('faithfulness', 0.0),
                  scores.get('answer_relevancy', 0.0),
                  scores.get('context_precision', 0.0),
                  scores.get('context_recall', 0.0),
                  scores.get('context_degradation', 0.0),
                  scores.get('failure_mode', None),
                  scores.get('chunk_efficiency', 0.0)))
            conn.commit()
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE session_id = ? 
                ORDER BY turn_number ASC
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_evaluation_results(self, model_name: str = None) -> List[Dict]:
        """Get evaluation results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if model_name:
                cursor.execute("SELECT * FROM evaluations WHERE model_name = ?", (model_name,))
            else:
                cursor.execute("SELECT * FROM evaluations")
            return [dict(row) for row in cursor.fetchall()]
    
    def add_document_record(self, filename: str, file_type: str, chunk_count: int) -> str:
        """Record uploaded document"""
        document_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (document_id, filename, file_type, chunk_count)
                VALUES (?, ?, ?, ?)
            """, (document_id, filename, file_type, chunk_count))
            conn.commit()
        return document_id
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all documents"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename, file_type, chunk_count, uploaded_at 
                FROM documents 
                ORDER BY uploaded_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def remove_document_record(self, filename: str):
        """Remove document record from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()
    
    def clear_all_documents(self):
        """Clear all document records"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents")
            conn.commit()
    
    def reset_all_evaluations(self):
        """Reset ALL evaluation and conversation data - preserve documents"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete in correct order (respecting foreign keys)
            cursor.execute("DELETE FROM evaluations")
            cursor.execute("DELETE FROM conversations") 
            cursor.execute("DELETE FROM sessions")
            cursor.execute("DELETE FROM model_performance")
            
            conn.commit()
            print("✅ All evaluation data cleared. Documents preserved.")