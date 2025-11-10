# ============================================================================
# models/vector_store.py - FIXED for multi-document retrieval
# ============================================================================
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to vector store"""
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc['content'])
            metadatas.append({
                'document_id': doc.get('document_id', ''),
                'filename': doc.get('filename', ''),
                'chunk_index': doc.get('chunk_index', 0),
                'chunk_size': len(doc['content'])
            })
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return ids
    
    def similarity_search(self, query: str, k: int = 30) -> List[Dict[str, Any]]:
        """
        FIXED: Guaranteed multi-document retrieval
        ENSURES at least 5 chunks from EVERY document
        """
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Get MAXIMUM possible chunks
        total_chunks = self.collection.count()
        
        all_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(total_chunks, 500),  # Get up to 500 chunks
            include=['documents', 'metadatas', 'distances']
        )
        
        if not all_results['documents'] or not all_results['documents'][0]:
            return []
        
        # Group ALL chunks by document
        chunks_by_document = {}
        
        for i in range(len(all_results['documents'][0])):
            filename = all_results['metadatas'][0][i].get('filename', 'unknown')
            similarity_score = float(1 - all_results['distances'][0][i])
            
            chunk_data = {
                'content': all_results['documents'][0][i],
                'metadata': all_results['metadatas'][0][i],
                'similarity_score': similarity_score,
                'filename': filename
            }
            
            if filename not in chunks_by_document:
                chunks_by_document[filename] = []
            
            chunks_by_document[filename].append(chunk_data)
        
        # Sort chunks within each document by similarity
        for doc_name in chunks_by_document:
            chunks_by_document[doc_name].sort(key=lambda x: x['similarity_score'], reverse=True)
        
        num_documents = len(chunks_by_document)
        
        print(f"\n{'='*60}")
        print(f"[RETRIEVAL DEBUG] Query: {query[:50]}...")
        print(f"[RETRIEVAL DEBUG] Found {num_documents} unique documents")
        for doc_name, chunks in chunks_by_document.items():
            print(f"  📄 {doc_name}: {len(chunks)} chunks, best score: {chunks[0]['similarity_score']:.3f}")
        print(f"{'='*60}\n")
        
        # MANDATORY: Take AT LEAST 7 chunks from EACH document
        final_chunks = []
        min_per_doc = 7  # INCREASED from 5 to 7
        
        if num_documents > 1:
            print(f"[RETRIEVAL] Multi-document strategy: {min_per_doc} chunks minimum per document")
            
            # PHASE 1: MANDATORY - Get minimum from EACH document
            for doc_name, chunks in chunks_by_document.items():
                selected = chunks[:min_per_doc]
                final_chunks.extend(selected)
                print(f"  ✓ Took {len(selected)} mandatory chunks from {doc_name}")
            
            # PHASE 2: Fill remaining with best overall
            remaining_slots = k - len(final_chunks)
            if remaining_slots > 0:
                all_remaining = []
                for doc_name, chunks in chunks_by_document.items():
                    all_remaining.extend(chunks[min_per_doc:])
                
                all_remaining.sort(key=lambda x: x['similarity_score'], reverse=True)
                additional = all_remaining[:remaining_slots]
                final_chunks.extend(additional)
                print(f"  + Added {len(additional)} additional best chunks")
            
            # Sort by similarity
            final_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Verify document diversity
            doc_counts = {}
            for chunk in final_chunks:
                doc = chunk['filename']
                doc_counts[doc] = doc_counts.get(doc, 0) + 1
            
            print(f"\n[FINAL SELECTION] {len(final_chunks)} chunks distributed:")
            for doc, count in doc_counts.items():
                print(f"  📊 {doc}: {count} chunks")
            print(f"{'='*60}\n")
            
            return final_chunks[:k]
        else:
            # Single document
            print(f"[RETRIEVAL] Single document found")
            all_chunks = list(chunks_by_document.values())[0]
            return all_chunks[:k]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }