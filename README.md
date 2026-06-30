# RAG Evaluation System: Multi-Document LLM Comparison

A comprehensive evaluation framework comparing GPT-3.5-turbo, Claude-3-Haiku, and Gemini-Pro in Retrieval-Augmented Generation (RAG) scenarios with multi-document synthesis.

## Project Overview

This system evaluates how well different LLMs handle complex queries requiring information synthesis from multiple enterprise documents. We tested three commercial models on 10 queries using Stanford AI Index 2025 and McKinsey State of AI reports.

## Key Findings

| Model | Faithfulness | Answer Coverage | Behavior |
|-------|-------------|-----------------|----------|
| **Claude-3-Haiku** | 0.904 ± 0.110 | 60% | Conservative, honest refusals |
| **Gemini-Pro** | 0.846 ± 0.104 | 100% | Comprehensive answering |
| **GPT-3.5-turbo** | 0.890 ± 0.121 | 100% | Balanced synthesis |

**Statistical Significance**: Chunk efficiency differences (p < 0.05, Cohen's d > 0.96) favor Claude/Gemini over GPT-3.5.

## Novel Metrics

- **Context Degradation**: Performance decay across conversation turns
- **Answer Coverage**: Distinguishes substantive answers from honest refusals
- **Chunk Efficiency**: How well models use relevant vs. irrelevant context
- **Faithfulness**: Semantic grounding in retrieved documents
- **Answer Relevancy**: Query-response alignment

## Architecture

```
Documents (PDF/DOCX) 
    → Chunking (250 chars, 16% overlap)
    → Vector Store (ChromaDB + sentence-transformers)
    → Retrieval (Top-30 chunks, multi-doc diversity)
    → LLM Generation (GPT-3.5 / Claude / Gemini)
    → Evaluation (5 metrics + statistical tests)
    → Results (IEEE-ready tables/figures)
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 3. Upload Documents
```bash
# Start backend
python main.py

# Upload via API (curl or Postman)
curl -X POST http://localhost:8000/api/v1/upload-documents \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### 4. Run Queries
```python
# Ask questions via API
POST /api/v1/chat
{
  "message": "What are AI adoption trends?",
  "session_id": "test-session",
  "model_name": "claude-3-haiku-20240307"
}
```

### 5. Generate Results
```bash
python generate_paper_results.py
```

Outputs IEEE-ready tables, figures (PNG/PDF), and LaTeX tables in `publication_results/run_<timestamp>/`

## Project Structure

```
├── main.py                     # FastAPI backend
├── config.py                   # Model configurations
├── models/
│   ├── evaluator.py           # Fair evaluation metrics
│   ├── llm_providers.py       # OpenAI/Claude/Gemini API wrappers
│   ├── vector_store.py        # Multi-doc retrieval (ChromaDB)
│   ├── document_processor.py  # PDF/DOCX chunking
│   └── database.py            # SQLite storage
├── generate_paper_results.py  # Publication-ready visualizations
└── requirements.txt
```

## 📈 Key Results

### Faithfulness vs. Answer Coverage Trade-off
- **Claude**: Highest accuracy, but refuses 40% when context insufficient
- **GPT-3.5/Gemini**: Always attempt answers, comparable accuracy

### Statistical Tests
- Chunk efficiency: **p = 0.033** (Claude vs GPT-3.5)
- Effect size: **Cohen's d = 1.033** (large)
- Faithfulness differences: **not significant** (p > 0.45)

### Practical Implications
**Use Claude for**: High-stakes (legal, medical, compliance)  
**Use Gemini/GPT-3.5 for**: Exploratory research, synthesis tasks  
**Honest refusals are features**: Enable error handling, prevent hallucinations

## Evaluation Methodology

**Documents**: Stanford AI Index 2025 (127 pages) + McKinsey State of AI (45 pages)  
**Queries**: 10 complex, multi-document synthesis questions  
**Retrieval**: 30 chunks per query (minimum 7 per document)  
**Metrics**: Faithfulness, Relevancy, Context Degradation, Coverage, Efficiency  
**Statistics**: Independent t-tests, Cohen's d effect sizes, 95% CI

## Generated Outputs

- `table1_comprehensive_metrics.csv` - All metrics with confidence intervals
- `table2_significance_tests.csv` - Statistical comparisons
- `figure1_metrics_comparison.pdf` - Bar chart across all metrics
- `figure2_radar_chart.pdf` - Multi-dimensional performance
- `figure3_context_degradation.pdf` - Performance over turns
- `figure4_answer_coverage.pdf` - Response type distribution
- `table_comparison.pdf/.png` - Clean comparison table
- `table1_latex.txt` - Ready for IEEE papers

## Research Paper

Full 8-page IEEE conference paper available with:
- 3 evaluation algorithms
- 2 comprehensive tables
- 5 publication-quality figures
- 20 academic references
- Statistical significance analysis

## Technologies

- **Backend**: FastAPI, Python 3.9+
- **LLMs**: OpenAI API, Anthropic API, Google Gemini API
- **Vector DB**: ChromaDB with sentence-transformers
- **Embeddings**: all-MiniLM-L6-v2
- **Evaluation**: scikit-learn, scipy, numpy
- **Visualization**: matplotlib, seaborn (IEEE-compliant)
- **Storage**: SQLite

## Key Innovation

**Fair Evaluation Framework**: Unlike traditional metrics that penalize all non-answers, our system distinguishes:
- **Honest refusals** (0.95 faithfulness) - "Context doesn't contain this"
- **Vague refusals** (0.60 faithfulness) - "I don't know"
- **Hallucinations** (0.20 faithfulness) - Making things up

This enables proper assessment of epistemic humility in RAG systems.

## Authors

Research project by Jash Ladhani & Manav Bhuta

## Acknowledgments

- Stanford HAI for AI Index Report 2025
- McKinsey & Company for State of AI Report
- OpenAI, Anthropic, and Google for API access
```

This README is concise, professional, and highlights your key contributions without being overwhelming. It includes all essential information for someone to understand and reproduce your work!
