import openai
import google.generativeai as genai
from anthropic import Anthropic
import asyncio
import time
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import os

class LLMProvider:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_providers = []
        self.setup_providers()
    
    def setup_providers(self):
        """Initialize available LLM providers"""
        # Google Gemini (FREE tier available!)
        if self.config.get('GOOGLE_API_KEY'):
            try:
                genai.configure(api_key=self.config['GOOGLE_API_KEY'])
                self.available_providers.append('google')
                print("✓ Google Gemini initialized")
            except Exception as e:
                print(f"Could not initialize Gemini: {e}")
        
        # Anthropic Claude (Generous free tier)
        if self.config.get('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
                self.available_providers.append('anthropic')
                print("✓ Anthropic Claude initialized")
            except Exception as e:
                print(f"Could not initialize Claude: {e}")
        
        # OpenAI (Optional - requires payment)
        if self.config.get('OPENAI_API_KEY'):
            try:
                openai.api_key = self.config['OPENAI_API_KEY']
                self.available_providers.append('openai')
                print("✓ OpenAI initialized")
            except Exception as e:
                print(f"Could not initialize OpenAI: {e}")
        
        if not self.available_providers:
            print("WARNING: No LLM providers initialized!")
            print("   Please add at least one API key to .env file")
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        if model_name.startswith('gpt'):
            return 'openai' in self.available_providers
        elif model_name.startswith('gemini'):
            return 'google' in self.available_providers
        elif model_name.startswith('claude'):
            return 'anthropic' in self.available_providers
        return False
    
    async def generate_response(self, model_name: str, messages: List[Dict[str, str]], 
                               context_chunks: List[str]) -> Dict[str, Any]:
        """Generate response from specified model"""
        
        # Check if model is available
        if not self.is_model_available(model_name):
            return {
                'response': f"Error: {model_name} is not available. Please configure the API key in .env file.",
                'token_count': 0,
                'response_time': 0,
                'model_name': model_name,
                'success': False,
                'error': 'Model not configured'
            }
        
        start_time = time.time()
        if model_name.startswith('claude'):
            raw_context = "\n\n".join(context_chunks)
            claude_start = time.time()
            claude_data = await self._claude_generate(model_name, raw_context, messages)
            response_time = time.time() - claude_start
            return {
                "success": True,
                "response": claude_data.get("content", ""),
                "token_count": claude_data.get("token_count", 0),
                "response_time": response_time,
                "model_name": model_name
            }

        # Prepare context
        context = "\n\n".join(context_chunks) if context_chunks else ""
        
        # Create prompt with context
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer questions accurately.
        
Context:
{context}

Instructions:
1. Base your answers on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Maintain conversation continuity by referencing previous exchanges when relevant
4. Be concise but comprehensive in your responses"""
        
        try:
            if model_name.startswith('gpt'):
                response = await self._openai_generate(model_name, system_prompt, messages)
            elif model_name.startswith('gemini'):
                response = await self._gemini_generate(model_name, system_prompt, messages)
            elif model_name.startswith('claude'):
                response = await self._claude_generate(model_name, system_prompt, messages)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            response_time = time.time() - start_time
            
            return {
                'response': response['content'],
                'token_count': response.get('token_count', 0),
                'response_time': response_time,
                'model_name': model_name,
                'success': True
            }
        
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'token_count': 0,
                'response_time': time.time() - start_time,
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    async def _openai_generate(self, model_name: str, system_prompt: str, 
                              messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate response using OpenAI models"""
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=1000,
            )
        
        return {
            'content': response.choices[0].message.content,
            'token_count': response.usage.total_tokens
        }
    
    async def _gemini_generate(self, model_name: str, system_prompt: str,
                              messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate response using Google Gemini"""
        # Use gemini-2.5-pro which is stable
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Format conversation for Gemini
        conversation_text = system_prompt + "\n\n"
        for msg in messages[-5:]:  # Last 5 messages for context
            role = "Human" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Use synchronous call (Gemini SDK doesn't support async properly)
        response = model.generate_content(conversation_text)
        
        return {
            'content': response.text,
            'token_count': len(response.text.split()) * 1.3  # Rough estimation
        }
    
    async def _claude_generate(self, model_name: str, system_prompt: str,
                          messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate response using Anthropic Claude (corrected for context usage and variable safety).
        """
        # Normalize model name
        if model_name == "claude-3-haiku":
            model_name = "claude-3-haiku-20240307"

        # System instruction (remains stable)
        system_text = (
            "You are an expert RAG assistant. "
            "Use the provided context as evidence to answer questions clearly and insightfully. "
            "If the context lacks direct facts, infer reasonable conclusions or summarize related insights. "
            "Only say the context lacks information if it is completely unrelated to the question."
        )

        # Safely extract user query (last message or fallback)
        user_query = ""
        if messages and isinstance(messages[-1], dict):
            user_query = messages[-1].get("content", "")
        elif isinstance(messages, list) and len(messages) > 0:
            user_query = str(messages[-1])
        else:
            user_query = "No question provided."

        # Combine context (system_prompt holds raw context text)
        context_text = system_prompt.strip() if system_prompt else ""

        # Merge context + question into one message
        full_prompt = f"Context:\n{context_text}\n\nNow, answer this question:\n{user_query}"

        conversation = [{"role": "user", "content": full_prompt}]

        # Call Anthropic API
        try:
            response = self.anthropic_client.messages.create(
                model=model_name,
                system=system_text,
                messages=conversation,
                max_tokens=1200,
                temperature=0.7
            )

            # Extract the actual response text
            content_text = ''.join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

            return {
                "content": content_text.strip(),
                "token_count": getattr(response.usage, "output_tokens", 0)
}

        except Exception as e:
            return {
                "content": f"Error generating response: {str(e)}",
                "token_count": 0,
                "response_time": 0
            }  


