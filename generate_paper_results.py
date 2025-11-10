"""
IEEE-Ready Publication Results Generator
Generates fair comparisons and publication-quality visualizations for IEEE papers
Usage: python generate_publication_results.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from datetime import datetime

# IEEE publication styling
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for IEEE
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 12

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class IEEEPublicationGenerator:
    def __init__(self, db_path="rag_evaluation.db"):
        self.db_path = db_path
        self.output_dir = "publication_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # DPI for publication quality
        self.dpi = 300
    
    def save_figure(self, filename_base):
        """Save figure in both PNG and PDF formats"""
        png_file = os.path.join(self.run_dir, f"{filename_base}.png")
        pdf_file = os.path.join(self.run_dir, f"{filename_base}.pdf")
        
        plt.savefig(png_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
        
        print(f"✅ Saved: {filename_base}.png and {filename_base}.pdf")
        plt.close()
    
    def load_and_clean_data(self):
        """Load data with robust cleaning"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT e.*, c.turn_number, c.model_used, c.session_id, c.response_time,
               c.user_message, c.assistant_response
        FROM evaluations e
        JOIN conversations c ON e.conversation_id = c.conversation_id
        ORDER BY c.session_id, c.turn_number
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Clean numeric columns
        numeric_cols = [
            'faithfulness_score', 'answer_relevancy', 'context_precision',
            'context_recall', 'chunk_efficiency', 'context_degradation_score',
            'response_time', 'turn_number'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN metrics
        df = df.dropna(subset=['faithfulness_score', 'answer_relevancy'], how='all')
        
        self.eval_df = df
        
        print(f"\n{'='*80}")
        print(f"DATA LOADED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Total records: {len(df)}")
        print(f"Models: {df['model_name'].unique()}")
        print(f"Sessions: {df['session_id'].nunique()}")
        print(f"Turn range: {df['turn_number'].min()} to {df['turn_number'].max()}")
        print(f"{'='*80}\n")
        
        return df
    
    def generate_table1_comprehensive_metrics(self):
        """Table 1: Comprehensive Performance Metrics (IEEE format)"""
        print("\n" + "="*80)
        print("TABLE 1: Comprehensive Model Performance")
        print("="*80)
        
        metrics = {
            'faithfulness_score': 'Faithfulness',
            'answer_relevancy': 'Answer Relevancy',
            'context_precision': 'Context Precision',
            'context_recall': 'Context Recall',
            'chunk_efficiency': 'Chunk Efficiency'
        }
        
        results = []
        
        for model in sorted(self.eval_df['model_name'].unique()):
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            
            row = {'Model': model, 'N': len(model_data)}
            
            for metric_col, metric_name in metrics.items():
                if metric_col in model_data.columns:
                    values = model_data[metric_col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        ci_95 = 1.96 * std_val / np.sqrt(len(values))
                        row[metric_name] = f"{mean_val:.3f} ± {ci_95:.3f}"
                        row[f"{metric_name}_mean"] = mean_val
                        row[f"{metric_name}_std"] = std_val
                    else:
                        row[metric_name] = "N/A"
            
            # Response time
            rt = model_data['response_time'].dropna()
            if len(rt) > 0:
                row['Avg Response Time (s)'] = f"{rt.mean():.2f} ± {rt.std():.2f}"
            
            # Answer coverage
            substantive_answers = 0
            total_responses = len(model_data)
            
            for _, response in model_data.iterrows():
                resp_text = str(response.get('assistant_response', '')).lower()
                if len(resp_text) > 50 and 'there is no information' not in resp_text:
                    substantive_answers += 1
            
            row['Answer Rate (%)'] = f"{(substantive_answers / total_responses * 100):.1f}%"
            
            results.append(row)
        
        df_table = pd.DataFrame(results)
        
        # Display
        display_cols = ['Model', 'N', 'Faithfulness', 'Answer Relevancy', 
                       'Context Precision', 'Context Recall', 'Answer Rate (%)']
        print("\n", df_table[display_cols].to_string(index=False))
        
        # Save
        output_file = os.path.join(self.run_dir, "table1_comprehensive_metrics.csv")
        df_table.to_csv(output_file, index=False)
        
        # Also save LaTeX-ready format
        latex_file = os.path.join(self.run_dir, "table1_latex.txt")
        with open(latex_file, 'w') as f:
            f.write("% IEEE LaTeX Table Format\n")
            f.write("% Copy this into your paper\n\n")
            f.write("\\begin{table}[!t]\n")
            f.write("\\caption{Comprehensive Performance Metrics Across Models}\n")
            f.write("\\label{table:performance}\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\hline\n")
            f.write("Model & N & Faithfulness & Relevancy & Precision & Recall & Answer Rate \\\\\n")
            f.write("\\hline\n")
            for _, row in df_table.iterrows():
                model_short = row['Model'].replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
                f.write(f"{model_short} & {row['N']} & {row['Faithfulness']} & {row['Answer Relevancy']} & {row['Context Precision']} & {row['Context Recall']} & {row['Answer Rate (%)']} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"\n✅ Saved to: {output_file}")
        print(f"✅ LaTeX format: {latex_file}")
        
        return df_table
    
    def generate_table2_statistical_tests(self):
        """Table 2: Statistical Significance Tests"""
        print("\n" + "="*80)
        print("TABLE 2: Statistical Significance Tests")
        print("="*80)
        
        models = sorted(self.eval_df['model_name'].unique())
        metrics = ['faithfulness_score', 'answer_relevancy', 'context_precision', 
                  'context_recall', 'chunk_efficiency']
        
        results = []
        
        for metric in metrics:
            if metric not in self.eval_df.columns:
                continue
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    data1 = self.eval_df[self.eval_df['model_name'] == model1][metric].dropna()
                    data2 = self.eval_df[self.eval_df['model_name'] == model2][metric].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        # T-test
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        
                        # Effect size (Cohen's d)
                        mean_diff = data1.mean() - data2.mean()
                        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
                        
                        # Effect size interpretation
                        if abs(cohens_d) < 0.2:
                            effect_label = "negligible"
                        elif abs(cohens_d) < 0.5:
                            effect_label = "small"
                        elif abs(cohens_d) < 0.8:
                            effect_label = "medium"
                        else:
                            effect_label = "large"
                        
                        results.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Comparison': f"{model1} vs {model2}",
                            'Mean Diff': f"{mean_diff:.3f}",
                            't-statistic': f"{t_stat:.3f}",
                            'p-value': f"{p_value:.4f}",
                            "Cohen's d": f"{cohens_d:.3f}",
                            'Effect Size': effect_label,
                            'Significant': "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        })
        
        if results:
            df_table = pd.DataFrame(results)
            print("\n", df_table.to_string(index=False))
            print("\n*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
            
            output_file = os.path.join(self.run_dir, "table2_significance_tests.csv")
            df_table.to_csv(output_file, index=False)
            print(f"\n✅ Saved to: {output_file}")
            
            return df_table
        else:
            print("\n⚠️  No sufficient data for statistical tests")
            return None
    
    def generate_figure1_metrics_comparison(self):
        """Figure 1: Bar Chart of All Metrics (IEEE Column Width)"""
        print("\n" + "="*80)
        print("FIGURE 1: Comprehensive Metrics Comparison")
        print("="*80)
        
        metrics = ['faithfulness_score', 'answer_relevancy', 'context_precision', 
                  'context_recall', 'chunk_efficiency']
        labels = ['Faithfulness', 'Relevancy', 'Precision', 'Recall', 'Efficiency']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.eval_df.columns]
        available_labels = [labels[i] for i, m in enumerate(metrics) if m in self.eval_df.columns]
        
        if len(available_metrics) < 3:
            print("⚠️  Not enough metrics for comparison")
            return
        
        # Calculate means and CIs
        models = sorted(self.eval_df['model_name'].unique())
        n_metrics = len(available_metrics)
        n_models = len(models)
        
        # IEEE single column width: 3.5 inches
        fig, ax = plt.subplots(figsize=(7, 4))
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for idx, model in enumerate(models):
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            
            means = []
            errors = []
            
            for metric in available_metrics:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    ci_95 = 1.96 * values.std() / np.sqrt(len(values))
                    means.append(mean_val)
                    errors.append(ci_95)
                else:
                    means.append(0)
                    errors.append(0)
            
            offset = (idx - n_models/2 + 0.5) * width
            model_label = model.replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
            
            ax.bar(x + offset, means, width, label=model_label, 
                  color=colors[idx], yerr=errors, capsize=3, 
                  error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
        
        ax.set_xlabel('Performance Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison Across All Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(available_labels, rotation=0)
        ax.set_ylim([0, 1.1])
        ax.legend(frameon=True, shadow=False, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        self.save_figure("figure1_metrics_comparison")
    
    def generate_figure2_radar_chart(self):
        """Figure 2: Radar Chart for Multi-Metric View"""
        print("\n" + "="*80)
        print("FIGURE 2: Multi-Metric Radar Chart")
        print("="*80)
        
        metrics = ['faithfulness_score', 'answer_relevancy', 'context_precision', 
                  'context_recall', 'chunk_efficiency']
        labels = ['Faithfulness', 'Relevancy', 'Precision', 'Recall', 'Efficiency']
        
        available_metrics = [m for m in metrics if m in self.eval_df.columns]
        available_labels = [labels[i] for i, m in enumerate(metrics) if m in self.eval_df.columns]
        
        if len(available_metrics) < 3:
            print("⚠️  Not enough metrics for radar chart")
            return
        
        # Create figure (IEEE double column width: 7 inches)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        models = sorted(self.eval_df['model_name'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            
            values = [model_data[metric].mean() for metric in available_metrics]
            values += values[:1]
            
            model_label = model.replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_label, 
                   color=colors[idx], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.set_title('Multi-Dimensional Performance Analysis', 
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), 
                 frameon=True, shadow=False)
        
        plt.tight_layout()
        self.save_figure("figure2_radar_chart")
    
    def generate_figure3_context_degradation_FIXED(self):
        """Figure 3: Context Degradation - FIXED to show actual data"""
        print("\n" + "="*80)
        print("FIGURE 3: Context Degradation Analysis (FIXED)")
        print("="*80)
        
        if 'context_degradation_score' not in self.eval_df.columns:
            print("⚠️  Context degradation data not available")
            return
        
        # Check if we have any actual degradation scores
        deg_scores = self.eval_df['context_degradation_score'].dropna()
        if len(deg_scores) == 0:
            print("⚠️  No context degradation scores available")
            return
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        models = sorted(self.eval_df['model_name'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        has_data = False
        
        for idx, model in enumerate(models):
            model_data = self.eval_df[self.eval_df['model_name'] == model].copy()
            
            # Remove rows with NaN degradation scores
            model_data = model_data.dropna(subset=['context_degradation_score', 'turn_number'])
            
            if len(model_data) < 2:
                print(f"  ⚠️ {model}: Insufficient data (only {len(model_data)} points)")
                continue
            
            # Group by turn
            turn_stats = model_data.groupby('turn_number')['context_degradation_score'].agg(['mean', 'std', 'count'])
            turn_stats = turn_stats[turn_stats['count'] >= 1]  # At least 1 sample
            
            if len(turn_stats) < 2:
                print(f"  ⚠️ {model}: Not enough turns ({len(turn_stats)})")
                continue
            
            turns = turn_stats.index.values
            means = turn_stats['mean'].values
            stds = turn_stats['std'].values
            counts = turn_stats['count'].values
            
            model_label = model.replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
            
            # Plot mean line
            ax.plot(turns, means, marker='o', linewidth=2, label=model_label, 
                   color=colors[idx], markersize=6, markeredgewidth=1.5, 
                   markeredgecolor='white')
            
            # Add confidence interval where we have multiple samples
            for i in range(len(turns)):
                if counts[i] > 1 and not np.isnan(stds[i]):
                    ci = 1.96 * stds[i] / np.sqrt(counts[i])
                    ax.errorbar(turns[i], means[i], yerr=ci, 
                              color=colors[idx], alpha=0.3, capsize=3,
                              capthick=1.5, elinewidth=1.5)
            
            has_data = True
            print(f"  ✓ {model}: {len(turns)} turns plotted")
        
        if not has_data:
            print("⚠️  No valid data to plot")
            plt.close()
            return
        
        ax.set_xlabel('Conversation Turn', fontweight='bold')
        ax.set_ylabel('Context Degradation Score', fontweight='bold')
        ax.set_title('Performance Across Conversation Turns', fontweight='bold')
        ax.legend(frameon=True, shadow=False, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_ylim([0, 1.05])
        
        # Add reference line
        ax.axhline(y=0.8, color='gray', linestyle=':', linewidth=1.5, 
                  alpha=0.5, label='Target (0.8)')
        
        plt.tight_layout()
        self.save_figure("figure3_context_degradation")
    
    def generate_figure4_answer_coverage(self):
        """Figure 4: Answer Coverage Comparison (NEW METRIC)"""
        print("\n" + "="*80)
        print("FIGURE 4: Answer Coverage Analysis")
        print("="*80)
        
        # Calculate answer coverage for each model
        coverage_data = []
        
        for model in self.eval_df['model_name'].unique():
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            
            total = len(model_data)
            substantive = 0
            honest_refusals = 0
            vague_refusals = 0
            
            for _, row in model_data.iterrows():
                resp = str(row.get('assistant_response', '')).lower()
                
                if 'there is no information' in resp or 'context does not' in resp:
                    honest_refusals += 1
                elif len(resp) > 50 and 'don\'t know' not in resp:
                    substantive += 1
                elif 'don\'t know' in resp or 'not sure' in resp:
                    vague_refusals += 1
            
            coverage_data.append({
                'Model': model,
                'Substantive Answers': (substantive / total * 100),
                'Honest Refusals': (honest_refusals / total * 100),
                'Vague Refusals': (vague_refusals / total * 100)
            })
        
        df_coverage = pd.DataFrame(coverage_data)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x = np.arange(len(df_coverage))
        width = 0.6
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        p1 = ax.bar(x, df_coverage['Substantive Answers'], width, 
                   label='Substantive Answers', color=colors[0], edgecolor='black')
        p2 = ax.bar(x, df_coverage['Honest Refusals'], width, 
                   bottom=df_coverage['Substantive Answers'],
                   label='Honest Refusals (Good)', color=colors[1], edgecolor='black')
        p3 = ax.bar(x, df_coverage['Vague Refusals'], width,
                   bottom=df_coverage['Substantive Answers'] + df_coverage['Honest Refusals'],
                   label='Vague Refusals (Poor)', color=colors[2], edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=13, weight='bold')
        ax.set_ylabel('Percentage of Responses (%)', fontsize=13, weight='bold')
        ax.set_title('Answer Coverage: How Models Respond to Questions', 
                    fontsize=15, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df_coverage['Model'], rotation=45, ha='right')
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.run_dir, "figure4_answer_coverage.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved to: {output_file}")
        plt.close()
    
    def generate_figure5_response_time(self):
        """Figure 5: Response Time Comparison"""
        print("\n" + "="*80)
        print("FIGURE 5: Response Time Analysis")
        print("="*80)
        
        if 'response_time' not in self.eval_df.columns:
            print("⚠️  Response time data not available")
            return
        
        fig, ax = plt.subplots(figsize=(7, 4))
        
        models = sorted(self.eval_df['model_name'].unique())
        response_times = []
        model_labels = []
        
        for model in models:
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            rt = model_data['response_time'].dropna()
            
            if len(rt) > 0:
                response_times.append(rt.values)
                model_label = model.replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
                model_labels.append(model_label)
        
        if not response_times:
            print("⚠️  No response time data available")
            plt.close()
            return
        
        # Box plot
        bp = ax.boxplot(response_times, labels=model_labels, patch_artist=True,
                       widths=0.6, showmeans=True, meanline=True)
        
        # Color boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(response_times)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Response Time (seconds)', fontweight='bold')
        ax.set_title('Response Time Distribution by Model', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        self.save_figure("figure5_response_time")
    
    def generate_summary_report(self):
        """Generate executive summary"""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        summary = []
        
        for model in sorted(self.eval_df['model_name'].unique()):
            model_data = self.eval_df[self.eval_df['model_name'] == model]
            
            model_label = model.replace('claude-3-haiku-20240307', 'Claude Haiku').replace('gpt-3.5-turbo', 'GPT-3.5').replace('gemini-pro', 'Gemini Pro')
            
            summary.append({
                'Model': model_label,
                'Total Responses': len(model_data),
                'Avg Faithfulness': model_data['faithfulness_score'].mean(),
                'Avg Relevancy': model_data['answer_relevancy'].mean(),
                'Avg Precision': model_data['context_precision'].mean(),
                'Avg Response Time': model_data['response_time'].mean(),
                'Sessions': model_data['session_id'].nunique()
            })
        
        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.round(3)
        
        print("\n", df_summary.to_string(index=False))
        
        output_file = os.path.join(self.run_dir, "executive_summary.csv")
        df_summary.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
    
    def generate_all(self):
        """Generate all results"""
        print("\n" + "="*80)
        print("IEEE PUBLICATION RESULTS GENERATOR")
        print("="*80)
        
        self.load_and_clean_data()
        
        if len(self.eval_df) == 0:
            print("\n❌ ERROR: No evaluation data found!")
            return
        
        print(f"\n📊 Generating IEEE-ready results...\n")
        
        # Tables
        self.generate_table1_comprehensive_metrics()
        self.generate_table2_statistical_tests()
        
        # Figures
        self.generate_figure1_metrics_comparison()
        self.generate_figure2_radar_chart()
        self.generate_figure3_context_degradation_FIXED()
        self.generate_figure4_answer_coverage()
        self.generate_figure5_response_time()
        
        # Summary
        self.generate_summary_report()

if __name__ == "__main__":
    generator = IEEEPublicationGenerator()
    generator.generate_all()