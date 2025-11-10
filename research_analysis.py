import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
from typing import Dict, Any, List

class ResearchAnalyzer:
    def __init__(self, db_path: str = "rag_evaluation.db"):
        self.db_path = db_path
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data from database"""
        conn = sqlite3.connect(self.db_path)
        
        tables = {
            'conversations': pd.read_sql_query("SELECT * FROM conversations", conn),
            'evaluations': pd.read_sql_query("SELECT * FROM evaluations", conn),
            'sessions': pd.read_sql_query("SELECT * FROM sessions", conn),
            'documents': pd.read_sql_query("SELECT * FROM documents", conn)
        }
        
        conn.close()
        return tables
    
    def analyze_context_degradation(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive context degradation analysis - FIXED"""
        conv_df = data['conversations']
        eval_df = data['evaluations']
        
        # Merge conversation and evaluation data
        merged_df = conv_df.merge(eval_df, on='conversation_id', how='left')
        
        results = {}
        
        # Analysis by model
        for model in merged_df['model_used'].unique():
            model_data = merged_df[merged_df['model_used'] == model].copy()
            
            # Group by session to analyze degradation curves
            degradation_curves = []
            for session_id in model_data['session_id'].unique():
                session_data = model_data[model_data['session_id'] == session_id].copy()
                session_data = session_data.sort_values('turn_number')
                
                if len(session_data) >= 5:  # Minimum turns for analysis
                    turns = session_data['turn_number'].values
                    scores = session_data['context_degradation_score'].values
                    
                    # Remove any NaN values
                    valid_indices = ~pd.isna(scores)
                    turns = turns[valid_indices]
                    scores = scores[valid_indices]
                    
                    if len(turns) < 3:
                        continue
                    
                    # Fit exponential decay curve
                    try:
                        from scipy.optimize import curve_fit
                        
                        def exp_decay(x, a, b, c):
                            return a * np.exp(-b * x) + c
                        
                        popt, pcov = curve_fit(exp_decay, turns, scores, maxfev=1000)
                        a, b, c = popt
                        
                        # Calculate R-squared
                        y_pred = exp_decay(turns, a, b, c)
                        ss_res = np.sum((scores - y_pred) ** 2)
                        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        degradation_curves.append({
                            'session_id': session_id,
                            'degradation_rate': float(b),
                            'initial_performance': float(a + c),
                            'asymptotic_performance': float(c),
                            'r_squared': float(r_squared),
                            'curve_quality': 'good' if r_squared > 0.7 else 'moderate'
                        })
                    except:
                        # Fallback to linear regression
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(turns, scores)
                            degradation_curves.append({
                                'session_id': session_id,
                                'degradation_rate': float(abs(slope)),
                                'initial_performance': float(intercept),
                                'linear_fit': True,
                                'r_squared': float(r_value ** 2)
                            })
                        except:
                            continue
            
            if degradation_curves:
                df_curves = pd.DataFrame(degradation_curves)
                results[model] = {
                    'mean_degradation_rate': float(df_curves['degradation_rate'].mean()),
                    'std_degradation_rate': float(df_curves['degradation_rate'].std()),
                    'mean_initial_performance': float(df_curves['initial_performance'].mean()),
                    'sessions_analyzed': len(degradation_curves),
                    'curves': degradation_curves
                }
        
        return results
    
    def analyze_failure_modes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze failure mode patterns"""
        eval_df = data['evaluations']
        
        failure_analysis = {}
        
        for model in eval_df['model_name'].unique():
            model_data = eval_df[eval_df['model_name'] == model]
            
            # Count failure modes
            failure_counts = model_data['failure_mode'].value_counts().to_dict()
            
            # Calculate failure rate
            total_responses = len(model_data)
            failures = total_responses - failure_counts.get('none', 0)
            failure_rate = failures / total_responses if total_responses > 0 else 0
            
            failure_analysis[model] = {
                'failure_rate': float(failure_rate),
                'failure_modes': {str(k): int(v) for k, v in failure_counts.items()},
                'total_responses': int(total_responses),
                'most_common_failure': str(max(failure_counts, key=failure_counts.get)) if failure_counts else 'none'
            }
        
        return failure_analysis
    
    def statistical_significance_tests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical significance tests between models - FIXED"""
        eval_df = data['evaluations']
        
        models = eval_df['model_name'].unique()
        metrics = ['faithfulness_score', 'answer_relevancy', 'context_precision', 
                  'context_recall', 'context_degradation_score', 'chunk_efficiency']
        
        results = {}
        
        for metric in metrics:
            results[metric] = {}
            
            # Perform pairwise t-tests
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    data1 = eval_df[eval_df['model_name'] == model1][metric].dropna()
                    data2 = eval_df[eval_df['model_name'] == model2][metric].dropna()
                    
                    if len(data1) > 0 and len(data2) > 0:
                        try:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            
                            # Calculate effect size
                            mean_diff = float(data1.mean() - data2.mean())
                            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                            effect_size = float(mean_diff / pooled_std) if pooled_std != 0 else 0
                            
                            results[metric][f"{model1}_vs_{model2}"] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05),
                                'mean_diff': mean_diff,
                                'effect_size': effect_size
                            }
                        except:
                            continue
        
        return results
    
    def generate_research_report(self):
        """Generate comprehensive research report"""
        data = self.load_data()
        
        print("="*80)
        print("RAG EVALUATION RESEARCH REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset Statistics
        print("1. DATASET STATISTICS")
        print("-"*80)
        print(f"Total Sessions: {len(data['sessions'])}")
        print(f"Total Conversations: {len(data['conversations'])}")
        print(f"Total Evaluations: {len(data['evaluations'])}")
        print(f"Documents Processed: {len(data['documents'])}")
        print()
        
        # Context Degradation Analysis
        print("2. CONTEXT DEGRADATION ANALYSIS")
        print("-"*80)
        try:
            degradation_results = self.analyze_context_degradation(data)
            
            for model, results in degradation_results.items():
                print(f"\n{model}:")
                print(f"  Mean Degradation Rate: {results['mean_degradation_rate']:.4f}")
                print(f"  Std Degradation Rate: {results['std_degradation_rate']:.4f}")
                print(f"  Mean Initial Performance: {results['mean_initial_performance']:.4f}")
                print(f"  Sessions Analyzed: {results['sessions_analyzed']}")
        except Exception as e:
            print(f"  Error in degradation analysis: {str(e)}")
        print()
        
        # Failure Mode Analysis
        print("3. FAILURE MODE ANALYSIS")
        print("-"*80)
        try:
            failure_results = self.analyze_failure_modes(data)
            
            for model, results in failure_results.items():
                print(f"\n{model}:")
                print(f"  Overall Failure Rate: {results['failure_rate']:.2%}")
                print(f"  Most Common Failure: {results['most_common_failure']}")
                print(f"  Failure Mode Distribution:")
                for mode, count in results['failure_modes'].items():
                    percentage = (count / results['total_responses']) * 100
                    print(f"    - {mode}: {count} ({percentage:.1f}%)")
        except Exception as e:
            print(f"  Error in failure analysis: {str(e)}")
        print()
        
        # Statistical Significance
        print("4. STATISTICAL SIGNIFICANCE TESTS")
        print("-"*80)
        try:
            significance_results = self.statistical_significance_tests(data)
            
            for metric, comparisons in significance_results.items():
                print(f"\n{metric}:")
                for comparison, stats_data in comparisons.items():
                    if stats_data['significant']:
                        print(f"  {comparison}:")
                        print(f"    p-value: {stats_data['p_value']:.4f} *")
                        print(f"    Mean Difference: {stats_data['mean_diff']:.4f}")
                        print(f"    Effect Size (Cohen's d): {stats_data['effect_size']:.4f}")
        except Exception as e:
            print(f"  Error in significance tests: {str(e)}")
        print()
        
        # Key Findings
        print("5. KEY RESEARCH FINDINGS")
        print("-"*80)
        
        try:
            eval_df = data['evaluations']
            
            if len(eval_df) == 0:
                print("  No evaluation data available")
            else:
                # Convert all numeric columns to proper float type
                numeric_cols = ['faithfulness_score', 'answer_relevancy', 'context_degradation_score']
                
                for col in numeric_cols:
                    eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce')
                
                # Remove rows with NaN values
                eval_df_clean = eval_df.dropna(subset=numeric_cols)
                
                if len(eval_df_clean) > 0:
                    # Group by model and calculate mean
                    avg_scores = eval_df_clean.groupby('model_name')[numeric_cols].mean()
                    
                    if not avg_scores.empty and len(avg_scores) > 0:
                        best_faithfulness = avg_scores['faithfulness_score'].idxmax()
                        best_relevancy = avg_scores['answer_relevancy'].idxmax()
                        best_degradation = avg_scores['context_degradation_score'].idxmax()
                        
                        print(f"Best Faithfulness: {best_faithfulness} ({avg_scores.loc[best_faithfulness, 'faithfulness_score']:.3f})")
                        print(f"Best Answer Relevancy: {best_relevancy} ({avg_scores.loc[best_relevancy, 'answer_relevancy']:.3f})")
                        print(f"Best Context Retention: {best_degradation} ({avg_scores.loc[best_degradation, 'context_degradation_score']:.3f})")
                    else:
                        print("  Insufficient data for model comparison")
                else:
                    print("  No valid numeric data for analysis")
        except Exception as e:
            print(f"  Error in key findings: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*80)
        print("END OF REPORT")
        print("="*80)
        
        return {
            'degradation_analysis': degradation_results if 'degradation_results' in locals() else {},
            'failure_analysis': failure_results if 'failure_results' in locals() else {},
            'significance_tests': significance_results if 'significance_results' in locals() else {},
            'dataset_stats': {
                'sessions': len(data['sessions']),
                'conversations': len(data['conversations']),
                'evaluations': len(data['evaluations']),
                'documents': len(data['documents'])
            }
        }

if __name__ == "__main__":
    analyzer = ResearchAnalyzer()
    report = analyzer.generate_research_report()
    
    # Save report to JSON
    with open('research_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nReport saved to research_report.json")