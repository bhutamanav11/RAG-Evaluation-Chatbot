"""
Fair Multi-Model Comparison Script
Automatically tests multiple models with SAME questions
Usage: python run_fair_comparison.py
"""

import requests
import time
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000/api/v1"

# Your standard questions
STANDARD_QUESTIONS = [
    "What is the company's total revenue?",
    "Where is the company headquarters located?",
    "Who is the current CEO?",
    "What are the main products or services?",
    "What is the profit margin?",
    "How many employees does the company have?",
    "What are the key business segments?",
    "What is the company's market capitalization?",
    "What are the major risks mentioned?",
    "What is the dividend policy?",
    "Who are the main competitors?",
    "What is the revenue growth rate?",
    "What are the company's strategic priorities?",
    "What is the debt-to-equity ratio?",
    "What are the geographic markets served?",
    "What is the R&D spending?",
    "What are the environmental initiatives?",
    "What is the customer base?",
    "What are the recent acquisitions?",
    "What is the future outlook?"
]

def create_session():
    """Create a new session"""
    response = requests.post(f"{API_BASE_URL}/sessions")
    return response.json()['session_id']

def ask_question(session_id, question, model_name):
    """Ask a single question"""
    chat_request = {
        "message": question,
        "session_id": session_id,
        "model_name": model_name
    }
    
    response = requests.post(f"{API_BASE_URL}/chat", json=chat_request)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error: {response.text}")
        return None

def run_model_comparison(models_to_test, questions):
    """Run fair comparison across multiple models"""
    
    print("="*80)
    print("FAIR MULTI-MODEL COMPARISON")
    print("="*80)
    print(f"\nModels to test: {', '.join(models_to_test)}")
    print(f"Questions: {len(questions)}")
    print("\n" + "="*80 + "\n")
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}\n")
        
        # Create new session for this model
        session_id = create_session()
        print(f"Session ID: {session_id}")
        
        model_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Asking: {question[:50]}...")
            
            result = ask_question(session_id, question, model_name)
            
            if result:
                scores = result['evaluation_scores']
                print(f"   ✓ Faithfulness: {scores.get('faithfulness', 0):.3f}")
                print(f"   ✓ Relevancy: {scores.get('answer_relevancy', 0):.3f}")
                print(f"   ✓ Response time: {result['response_time']:.2f}s")
                
                model_results.append({
                    'turn': i,
                    'question': question,
                    'response': result['response'],
                    'scores': scores,
                    'response_time': result['response_time']
                })
            else:
                print(f"   ✗ Failed")
            
            # Small delay to avoid rate limits
            time.sleep(2)
        
        results[model_name] = {
            'session_id': session_id,
            'results': model_results
        }
        
        print(f"\n✅ Completed {model_name}: {len(model_results)} questions answered")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comparison_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n### SUMMARY ###\n")
    for model_name, data in results.items():
        scores = [r['scores'] for r in data['results']]
        
        avg_faithfulness = sum(s.get('faithfulness', 0) for s in scores) / len(scores)
        avg_relevancy = sum(s.get('answer_relevancy', 0) for s in scores) / len(scores)
        avg_time = sum(r['response_time'] for r in data['results']) / len(data['results'])
        
        print(f"{model_name}:")
        print(f"  Avg Faithfulness: {avg_faithfulness:.3f}")
        print(f"  Avg Relevancy: {avg_relevancy:.3f}")
        print(f"  Avg Response Time: {avg_time:.2f}s")
        print()
    
    print("Now run: python generate_paper_results.py")
    print("="*80)

if __name__ == "__main__":
    # Check API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API is not running!")
            print("   Start it with: python main.py")
            exit(1)
        
        available_models = response.json().get('available_models', [])
        print(f"Available models: {available_models}")
        
    except:
        print("❌ Cannot connect to API!")
        print("   Make sure you're running: python main.py")
        exit(1)
    
    # Select models to test
    models_to_test = available_models  # Test all available
    
    # Or manually specify:
    # models_to_test = ["gemini-pro", "claude-3-haiku-20240307"]
    
    if not models_to_test:
        print("❌ No models available!")
        exit(1)
    
    # Confirm before starting
    print("\n⚠️  IMPORTANT:")
    print("   1. Make sure documents are uploaded in the system")
    print("   2. This will take ~10 minutes for 20 questions × 2 models")
    print(f"   3. Total API calls: {len(STANDARD_QUESTIONS) * len(models_to_test)}")
    print()
    
    confirm = input("Continue? (yes/no): ").lower()
    
    if confirm != 'yes':
        print("Cancelled.")
        exit(0)
    
    # Run comparison
    run_model_comparison(models_to_test, STANDARD_QUESTIONS)