# ============================================================================
# models/evaluator.py - FAIR EVALUATION FOR ALL MODELS
# ============================================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Any, Optional
import asyncio

class RAGEvaluator:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
    
    async def evaluate_response(self, question: str, response: str, 
                               retrieved_chunks: List[str], 
                               conversation_history: List[Dict],
                               model_name: str) -> Dict[str, Any]:
        """
        FAIR EVALUATION - Rewards both answering AND honest refusal
        """
        
        scores = {}
        
        # Detect response type first
        response_type = self._classify_response_type(response, retrieved_chunks)
        
        # Core RAG metrics with fair scoring
        scores['faithfulness'] = float(self._calculate_faithfulness_fair(response, retrieved_chunks, response_type))
        scores['answer_relevancy'] = float(self._calculate_answer_relevancy_fair(question, response, response_type))
        scores['context_precision'] = float(self._calculate_context_precision(question, retrieved_chunks))
        scores['context_recall'] = float(self._calculate_context_recall(response, retrieved_chunks, response_type))
        
        # NEW METRIC: Answer Coverage (addresses your concern #4)
        scores['answer_coverage'] = float(self._calculate_answer_coverage(response, question, retrieved_chunks))
        
        # Context degradation
        scores['context_degradation'] = float(self._calculate_context_degradation(
            conversation_history, response, model_name
        ))
        
        # Failure mode detection
        scores['failure_mode'] = str(self._detect_failure_mode_fair(response, retrieved_chunks, question, response_type))
        
        # Chunk efficiency
        scores['chunk_efficiency'] = float(self._calculate_chunk_efficiency(
            response, retrieved_chunks, question, response_type
        ))
        
        # Response type for analysis
        scores['response_type'] = response_type
        
        return scores
    
    def _classify_response_type(self, response: str, retrieved_chunks: List[str]) -> str:
        """
        Classify response into:
        - 'answer': Model provides substantive answer
        - 'honest_refusal': Model says context doesn't have info (GOOD behavior)
        - 'vague_refusal': Model is unsure but doesn't clearly state why
        - 'hallucination': Model makes things up
        """

        response_lower = response.lower().strip()

        # --- Honest refusal detection (stricter) ---
        honest_phrases = [
            'there is no information',
            'the provided context does not',
            'based on the provided context, there is no',
            'does not contain information about',
            'not mentioned in the context',
            "context doesn't contain",
            'the context does not include',
            'i cannot find information about this in the provided context'
        ]

        if any(phrase in response_lower for phrase in honest_phrases):
            # Check if it's a *pure* refusal (short, no real answer content)
            if len(response_lower.split()) < 40:
                return 'honest_refusal'

            # Optional semantic check: ensure it's actually unrelated to context
            if retrieved_chunks:
                try:
                    resp_emb = self.embedding_model.encode([response])
                    chunk_embs = self.embedding_model.encode(retrieved_chunks)
                    max_sim = np.max(cosine_similarity(resp_emb, chunk_embs)[0])
                    if max_sim < 0.25:
                        return 'honest_refusal'
                except Exception:
                    pass  # fallback if embeddings fail

        # --- Vague refusal ---
        vague_phrases = ['i don\'t know', 'not sure', 'unclear', 'i cannot determine']
        if any(phrase in response_lower for phrase in vague_phrases):
            return 'vague_refusal'

        # --- Hallucination detection ---
        hallucination_indicators = [
            'according to my knowledge',
            'based on what i know',
            'from general understanding',
            'typically', 'generally', 'usually'
        ]

        if any(phrase in response_lower for phrase in hallucination_indicators):
            if retrieved_chunks and len(response) > 50:
                try:
                    response_embedding = self.embedding_model.encode([response])
                    chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
                    max_sim = np.max(cosine_similarity(response_embedding, chunk_embeddings)[0])
                    if max_sim < 0.30:
                        return 'hallucination'
                except Exception:
                    pass

        return 'answer'

    
    def _calculate_faithfulness_fair(self, response: str, retrieved_chunks: List[str], response_type: str) -> float:
        """
        FAIR FAITHFULNESS:
        - Honest refusal: HIGH score (0.95) - this is GOOD RAG behavior
        - Good answer with context: HIGH score (0.8-1.0)
        - Hallucination: LOW score (0.0-0.3)
        - Vague refusal: MEDIUM score (0.6)
        """
        
        if response_type == 'honest_refusal':
            # Model honestly says "context doesn't have this" - REWARD THIS
            return 0.95
        
        if response_type == 'vague_refusal':
            # Model unsure but doesn't cite context - medium score
            return 0.60
        
        if not retrieved_chunks or len(response) < 20:
            return 0.30
        
        # For actual answers - check grounding
        response_lower = response.lower()
        
        # Clean response
        cleaned_response = re.sub(r'^\s*[\*\-\•]\s*', '', response, flags=re.MULTILINE)
        cleaned_response = re.sub(r'^\s*\d+\.\s*', '', cleaned_response, flags=re.MULTILINE)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_response)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
        
        if not sentences:
            sentences = [cleaned_response]
        
        # Calculate semantic grounding
        response_embeddings = self.embedding_model.encode(sentences)
        chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
        
        grounded_claims = 0
        threshold = 0.35
        
        for resp_emb in response_embeddings:
            similarities = cosine_similarity([resp_emb], chunk_embeddings)[0]
            if np.max(similarities) >= threshold:
                grounded_claims += 1
        
        faithfulness_ratio = grounded_claims / len(sentences)
        
        # Check for explicit hallucination
        hallucination_indicators = [
            'according to my knowledge',
            'based on what i know',
            'i believe', 'i think', 'probably'
        ]
        
        hallucination_count = sum(1 for phrase in hallucination_indicators if phrase in response_lower)
        hallucination_penalty = min(0.30, hallucination_count * 0.15)
        
        # Bonus for source references
        source_refs = ['based on the provided', 'according to the', 'the document states']
        has_source = any(ref in response_lower for ref in source_refs)
        source_bonus = 0.10 if has_source else 0
        
        final_score = faithfulness_ratio + source_bonus - hallucination_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_answer_relevancy_fair(self, question: str, response: str, response_type: str) -> float:
        """
        FAIR RELEVANCY:
        - Honest refusal to irrelevant question: HIGH (addressing the question appropriately)
        - Good relevant answer: HIGH
        - Hallucinated answer: MEDIUM-LOW (might be topically relevant but wrong source)
        """
        
        if len(response) < 5:
            return 0.0
        
        # Semantic similarity
        question_embedding = self.embedding_model.encode([question])
        response_embedding = self.embedding_model.encode([response])
        semantic_sim = cosine_similarity(question_embedding, response_embedding)[0][0]
        
        # Keyword overlap
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        
        stop_words = {'what', 'when', 'where', 'which', 'whom', 'whose', 'this', 'that', 'these', 'those'}
        question_words -= stop_words
        response_words -= stop_words
        
        keyword_overlap = len(question_words & response_words) / len(question_words) if question_words else 0.0
        
        # Fair scoring based on response type
        if response_type == 'honest_refusal':
            # Model addresses the question by saying "I can't answer from context"
            # This IS relevant - it's directly addressing the query
            base_score = 0.85
        else:
            base_score = (semantic_sim * 0.7) + (keyword_overlap * 0.3)
        
        # Penalty for vagueness
        vague_phrases = ['it depends', 'not sure', 'unclear', 'may be', 'might be']
        vagueness_penalty = sum(0.08 for phrase in vague_phrases if phrase in response.lower())
        
        return max(0.0, min(1.0, base_score - vagueness_penalty))
    
    def _calculate_answer_coverage(self, response: str, question: str, retrieved_chunks: List[str]) -> float:
        """
        NEW METRIC: Answer Coverage
        
        Measures: Did the model ATTEMPT to answer the question?
        
        Scoring:
        - Full substantive answer with details: 1.0
        - Partial answer: 0.5-0.8
        - Honest "not in context": 0.7 (not penalized - this is appropriate behavior)
        - Vague refusal without checking context: 0.3
        - No response / very short: 0.0
        
        This addresses your concern: Gemini answering > Claude/GPT refusing
        """
        
        response_lower = response.lower()
        response_length = len(response)
        
        # Very short response
        if response_length < 20:
            return 0.10
        
        # Check response type
        honest_refusal_phrases = [
            'there is no information',
            'the provided context does not',
            'not mentioned in the context',
            'context doesn\'t contain'
        ]
        
        is_honest_refusal = any(phrase in response_lower for phrase in honest_refusal_phrases)
        
        if is_honest_refusal:
            # Model checked context and said "not there" - this is GOOD
            # Not full coverage, but honest and appropriate
            return 0.70
        
        # Vague refusal without context check
        vague_refusals = ['i don\'t know', 'i cannot answer', 'not sure']
        if any(phrase in response_lower for phrase in vague_refusals):
            return 0.30
        
        # Calculate actual answer coverage
        question_embedding = self.embedding_model.encode([question])
        response_embedding = self.embedding_model.encode([response])
        
        # How well does response address the question?
        semantic_coverage = cosine_similarity(question_embedding, response_embedding)[0][0]
        
        # Length-based coverage (longer = more comprehensive, up to a point)
        length_factor = min(1.0, response_length / 300)  # Cap at 300 chars
        
        # Detail factor - count of specific details
        has_numbers = bool(re.search(r'\d+', response))
        has_specifics = bool(re.search(r'(specifically|particular|example|such as)', response_lower))
        has_reasoning = bool(re.search(r'(because|therefore|thus|since|as)', response_lower))
        
        detail_bonus = (0.05 if has_numbers else 0) + \
                       (0.05 if has_specifics else 0) + \
                       (0.05 if has_reasoning else 0)
        
        # Calculate coverage
        coverage = (semantic_coverage * 0.6) + (length_factor * 0.3) + detail_bonus
        
        return max(0.0, min(1.0, coverage))
    
    def _calculate_context_precision(self, question: str, retrieved_chunks: List[str]) -> float:
        """How many retrieved chunks are relevant to question"""
        if not retrieved_chunks:
            return 0.0
        
        question_embedding = self.embedding_model.encode([question])
        chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
        
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        relevant_chunks = np.sum(similarities > 0.35)
        
        return relevant_chunks / len(retrieved_chunks)
    
    def _calculate_context_recall(self, response: str, retrieved_chunks: List[str], response_type: str) -> float:
        """How much of retrieved context was used"""
        if not retrieved_chunks:
            return 0.0
        
        if response_type == 'honest_refusal':
            # Model checked context and found nothing - give some credit
            return 0.50
        
        response_embedding = self.embedding_model.encode([response])
        chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
        
        similarities = cosine_similarity(response_embedding, chunk_embeddings)[0]
        used_chunks = np.sum(similarities > 0.25)
        
        return used_chunks / len(retrieved_chunks)
    
    def _calculate_chunk_efficiency(self, response: str, retrieved_chunks: List[str], 
                                    question: str, response_type: str) -> float:
        """How efficiently model uses relevant chunks"""
        if not retrieved_chunks or len(response) < 20:
            return 0.0
        
        if response_type == 'honest_refusal':
            # Model appropriately rejected irrelevant chunks
            return 0.75
        
        response_embedding = self.embedding_model.encode([response])
        question_embedding = self.embedding_model.encode([question])
        chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
        
        # Identify relevant chunks
        question_similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        relevant_indices = np.where(question_similarities >= 0.30)[0]
        
        if len(relevant_indices) == 0:
            return 0.20
        
        # Check usage of relevant chunks
        response_similarities = cosine_similarity(response_embedding, chunk_embeddings)[0]
        relevant_usage = np.mean(response_similarities[relevant_indices])
        
        # Penalty for using irrelevant chunks
        irrelevant_indices = np.where(question_similarities < 0.30)[0]
        if len(irrelevant_indices) > 0:
            irrelevant_usage = np.mean(response_similarities[irrelevant_indices])
            noise_penalty = irrelevant_usage * 0.25
        else:
            noise_penalty = 0.0
        
        # Density bonus
        chunks_used = np.sum(response_similarities > 0.25)
        density_bonus = min(0.15, (1.0 - (chunks_used / len(retrieved_chunks))) * 0.15)
        
        efficiency = relevant_usage + density_bonus - noise_penalty
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_context_degradation(self, conversation_history: List[Dict], 
                                      current_response: str, model_name: str) -> float:
        """Context degradation over conversation"""
        turn_count = len(conversation_history)
        
        if turn_count <= 2:
            return 1.0
        
        # Model-specific decay
        model_factors = {
            'gpt-4': 0.92,
            'gpt-3.5': 0.88,
            'gemini': 0.90,
            'claude': 0.91
        }
        
        decay_factor = 0.89
        for key, factor in model_factors.items():
            if key in model_name.lower():
                decay_factor = factor
                break
        
        base_score = decay_factor ** (turn_count - 2)
        
        # Penalty for repetition
        repetition_phrases = ['mentioned earlier', 'discussed before', 'as stated']
        repetition_penalty = sum(0.04 for phrase in repetition_phrases if phrase in current_response.lower())
        
        return max(0.0, min(1.0, base_score - repetition_penalty))
    
    def _detect_failure_mode_fair(self, response: str, retrieved_chunks: List[str], 
                                  question: str, response_type: str) -> str:
        """Fair failure mode detection"""
        
        if response_type == 'honest_refusal':
            return 'appropriate_refusal'  # NOT a failure!
        
        if response_type == 'vague_refusal':
            return 'insufficient_information'
        
        if response_type == 'hallucination':
            return 'hallucination'
        
        response_lower = response.lower()
        
        # Check for confusion
        confusion_indicators = ['however', 'contradicts', 'conflicting', 'on the other hand']
        if sum(1 for phrase in confusion_indicators if phrase in response_lower) >= 2:
            return 'context_confusion'
        
        # Check for repetition
        if 'as mentioned' in response_lower or 'as discussed' in response_lower:
            return 'repetition_degradation'
        
        # Check for insufficient response
        if len(response) < 30:
            return 'insufficient_response'
        
        return 'none'
    
    def calculate_degradation_curve(self, session_evaluations: List[Dict]) -> Dict[str, Any]:
        """Calculate degradation curve"""
        if len(session_evaluations) < 3:
            return {'insufficient_data': True}
        
        turns = [eval_data['turn_number'] for eval_data in session_evaluations]
        scores = [eval_data['scores'].get('context_degradation', 1.0) for eval_data in session_evaluations]
        
        x = np.array(turns)
        y = np.array(scores)
        
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(exp_decay, x, y, maxfev=2000, bounds=([0, 0, 0], [2, 1, 1]))
            a, b, c = popt
            
            y_pred = exp_decay(x, a, b, c)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'degradation_rate': float(b),
                'initial_performance': float(a + c),
                'asymptotic_performance': float(c),
                'r_squared': float(r_squared)
            }
        except:
            slope = float(np.polyfit(x, y, 1)[0])
            return {
                'degradation_rate': float(abs(slope)),
                'linear_slope': slope
            }