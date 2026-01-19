"""
Research-Grade LLM Failure-Oriented Evaluation Analysis
Comprehensive statistical analysis with publication-quality visualizations
"""

import sys
import os

# Change to project root directory
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy import stats

# Set style for publication-quality plots
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.family'] = 'serif'


def load_results():
    """Load evaluation results from JSON files"""
    print("\n" + "="*80)
    print("📂 LOADING RESULTS")
    print("="*80)
    
    results_files = glob('results/metrics/*.json')
    
    # Separate model results from statistical analysis
    model_files = [f for f in results_files if 'statistical_analysis' not in f]
    stats_files = [f for f in results_files if 'statistical_analysis' in f]
    
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(model_files)} model result files")
    
    # Load model results
    all_results = []
    for file in model_files[:10]:  # Load latest results per model
        with open(file, 'r') as f:
            result = json.load(f)
            # Only keep latest result per model
            if not any(r['model'] == result['model'] for r in all_results):
                all_results.append(result)
    
    # Load statistical analysis if exists
    statistical_analysis = None
    if stats_files:
        stats_files.sort(key=os.path.getmtime, reverse=True)
        with open(stats_files[0], 'r') as f:
            statistical_analysis = json.load(f)
    
    print(f"\nLoaded results for models:")
    for r in all_results:
        print(f"  - {r['model']}")
    
    return all_results, statistical_analysis


def create_summary_dataframe(all_results):
    """Extract summary statistics into DataFrame"""
    print("\n" + "="*80)
    print("📊 CREATING SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    
    for result in all_results:
        model_name = result['model']
        summary = result['summary']
        
        row = {
            'Model': model_name,
            'Basic Consistency': summary['contradictory_contexts']['avg_consistency_opposing_contexts'],
            'Basic Hedging Rate': summary['contradictory_contexts']['hedging_rate'],
            'Variance': summary['response_variance']['avg_variance'],
            'Similarity': summary['response_variance']['avg_mean_similarity'],
        }
        
        # Add advanced metrics if available
        if 'advanced_evaluation' in summary:
            row['Advanced Consistency'] = summary['advanced_evaluation']['overall_consistency']
            row['Advanced Hedging Rate'] = summary['advanced_evaluation']['overall_hedging_rate']
            row['Total Test Cases'] = summary['advanced_evaluation']['total_cases']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    return summary_df


def plot_overall_comparison(summary_df):
    """Create comprehensive comparison charts"""
    print("\n" + "="*80)
    print("📈 GENERATING OVERALL COMPARISON CHARTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    # Plot 1: Basic vs Advanced Consistency
    ax1 = axes[0, 0]
    if 'Advanced Consistency' in summary_df.columns:
        ax1.bar(x - width/2, summary_df['Basic Consistency'], width, 
                label='Basic Tests', alpha=0.8, color='#3498db')
        ax1.bar(x + width/2, summary_df['Advanced Consistency'], width, 
                label='Advanced Tests', alpha=0.8, color='#2ecc71')
    else:
        ax1.bar(x, summary_df['Basic Consistency'], width, 
                label='Consistency', alpha=0.8, color='#3498db')
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Consistency Score', fontweight='bold')
    ax1.set_title('Consistency: Basic vs Advanced Tests', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Hedging Rates
    ax2 = axes[0, 1]
    if 'Advanced Hedging Rate' in summary_df.columns:
        ax2.bar(x - width/2, summary_df['Basic Hedging Rate'], width, 
                label='Basic Tests', alpha=0.8, color='#e74c3c')
        ax2.bar(x + width/2, summary_df['Advanced Hedging Rate'], width, 
                label='Advanced Tests', alpha=0.8, color='#e67e22')
    else:
        ax2.bar(x, summary_df['Basic Hedging Rate'], width, 
                label='Hedging Rate', alpha=0.8, color='#e74c3c')
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Hedging Rate', fontweight='bold')
    ax2.set_title('Hedging Behavior Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Variance and Similarity
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar(x - width/2, summary_df['Variance'] * 1000, width, 
                    label='Variance (×1000)', alpha=0.8, color='#9b59b6')
    bars2 = ax3_twin.bar(x + width/2, summary_df['Similarity'], width, 
                         label='Similarity', alpha=0.8, color='#1abc9c')
    
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Variance (×1000)', fontweight='bold', color='#9b59b6')
    ax3_twin.set_ylabel('Similarity Score', fontweight='bold', color='#1abc9c')
    ax3.set_title('Response Variance & Similarity', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax3.tick_params(axis='y', labelcolor='#9b59b6')
    ax3_twin.tick_params(axis='y', labelcolor='#1abc9c')
    ax3.grid(axis='y', alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 4: Composite Score (if advanced metrics available)
    ax4 = axes[1, 1]
    if 'Advanced Consistency' in summary_df.columns:
        # Calculate composite score
        composite = (summary_df['Advanced Consistency'] - 
                    0.3 * summary_df['Advanced Hedging Rate'] - 
                    10 * summary_df['Variance'])
        
        colors = plt.cm.RdYlGn(composite / composite.max())
        bars = ax4.barh(summary_df['Model'], composite, alpha=0.8, color=colors)
        
        ax4.set_xlabel('Composite Score', fontweight='bold')
        ax4.set_ylabel('Model', fontweight='bold')
        ax4.set_title('Overall Performance Ranking', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (model, score) in enumerate(zip(summary_df['Model'], composite)):
            ax4.text(score, i, f' {score:.3f}', va='center', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Advanced metrics not available', 
                ha='center', va='center', fontsize=12)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.axis('off')
    
    plt.tight_layout()
    
    filepath = 'results/visualizations/comprehensive_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def plot_category_breakdown(all_results):
    """Plot performance by category"""
    print("\n" + "="*80)
    print("📊 GENERATING CATEGORY BREAKDOWN")
    print("="*80)
    
    # Extract category data
    category_data = {}
    
    for result in all_results:
        model = result['model']
        
        if 'advanced_cases' not in result or not result['advanced_cases']:
            continue
        
        for case in result['advanced_cases']:
            category = case['category']
            
            if category not in category_data:
                category_data[category] = {}
            
            if model not in category_data[category]:
                category_data[category][model] = []
            
            category_data[category][model].append(case['consistency_a_vs_b'])
    
    if not category_data:
        print("⚠️  No category data available")
        return
    
    # Create subplot for each category
    categories = sorted(category_data.keys())
    n_cats = len(categories)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('Performance by Category', fontsize=18, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, category in enumerate(categories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        models = list(category_data[category].keys())
        means = [np.mean(category_data[category][m]) for m in models]
        stds = [np.std(category_data[category][m]) for m in models]
        
        x = np.arange(len(models))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Consistency Score', fontweight='bold')
        ax.set_title(category.replace('_', ' ').title(), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Hide empty subplots
    for idx in range(n_cats, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    filepath = 'results/visualizations/category_breakdown.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def plot_difficulty_analysis(all_results):
    """Plot performance by difficulty level"""
    print("\n" + "="*80)
    print("📊 GENERATING DIFFICULTY ANALYSIS")
    print("="*80)
    
    # Extract difficulty data
    difficulty_data = {}
    
    for result in all_results:
        model = result['model']
        
        if 'advanced_cases' not in result or not result['advanced_cases']:
            continue
        
        for case in result['advanced_cases']:
            difficulty = case.get('difficulty', 'medium')
            
            if difficulty not in difficulty_data:
                difficulty_data[difficulty] = {}
            
            if model not in difficulty_data[difficulty]:
                difficulty_data[difficulty][model] = []
            
            difficulty_data[difficulty][model].append(case['consistency_a_vs_b'])
    
    if not difficulty_data:
        print("⚠️  No difficulty data available")
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    difficulties = ['easy', 'medium', 'hard']
    difficulties = [d for d in difficulties if d in difficulty_data]
    
    models = list(set(m for d in difficulty_data.values() for m in d.keys()))
    
    x = np.arange(len(difficulties))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        means = []
        for diff in difficulties:
            if model in difficulty_data[diff]:
                means.append(np.mean(difficulty_data[diff][model]))
            else:
                means.append(0)
        
        offset = (i - len(models)/2) * width + width/2
        ax.bar(x + offset, means, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Difficulty Level', fontweight='bold', fontsize=12)
    ax.set_ylabel('Consistency Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance by Difficulty Level', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(title='Model', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    filepath = 'results/visualizations/difficulty_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def plot_statistical_significance(statistical_analysis):
    """Plot statistical significance heatmap"""
    if not statistical_analysis or 'pairwise_comparisons' not in statistical_analysis:
        print("\n⚠️  No statistical analysis available")
        return
    
    print("\n" + "="*80)
    print("📊 GENERATING STATISTICAL SIGNIFICANCE HEATMAP")
    print("="*80)
    
    comparisons = statistical_analysis['pairwise_comparisons']
    
    if not comparisons:
        print("⚠️  No pairwise comparisons available")
        return
    
    # Extract models
    models = list(set([c.split('_vs_')[0] for c in comparisons.keys()] + 
                     [c.split('_vs_')[1] for c in comparisons.keys()]))
    
    # Create matrices
    n = len(models)
    p_value_matrix = np.ones((n, n))
    effect_size_matrix = np.zeros((n, n))
    
    for comparison, data in comparisons.items():
        parts = comparison.split('_vs_')
        model_a, model_b = parts[0], parts[1]
        
        i = models.index(model_a)
        j = models.index(model_b)
        
        p_value_matrix[i, j] = data['t_test']['p_value']
        p_value_matrix[j, i] = data['t_test']['p_value']
        
        effect = data['effect_sizes']['cohens_d']
        effect_size_matrix[i, j] = effect
        effect_size_matrix[j, i] = -effect
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Statistical Analysis: Model Comparisons', fontsize=16, fontweight='bold')
    
    # P-value heatmap
    ax1 = axes[0]
    sns.heatmap(p_value_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=models, yticklabels=models, ax=ax1,
                cbar_kws={'label': 'p-value'}, vmin=0, vmax=0.1)
    ax1.set_title('Statistical Significance (p-values)', fontweight='bold')
    
    # Effect size heatmap
    ax2 = axes[1]
    sns.heatmap(effect_size_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                xticklabels=models, yticklabels=models, ax=ax2,
                cbar_kws={'label': "Cohen's d"}, center=0)
    ax2.set_title('Effect Sizes (Cohen\'s d)', fontweight='bold')
    
    plt.tight_layout()
    
    filepath = 'results/visualizations/statistical_significance.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def print_key_findings(summary_df, statistical_analysis):
    """Print key insights and findings"""
    print("\n" + "="*80)
    print("🔍 KEY FINDINGS")
    print("="*80)
    
    if 'Advanced Consistency' in summary_df.columns:
        best_consistency = summary_df.loc[summary_df['Advanced Consistency'].idxmax()]
        worst_variance = summary_df.loc[summary_df['Variance'].idxmax()]
        most_hedging = summary_df.loc[summary_df['Advanced Hedging Rate'].idxmax()]
        
        print(f"\n✅ Most Consistent Model: {best_consistency['Model']}")
        print(f"   Advanced Consistency: {best_consistency['Advanced Consistency']:.3f}")
        print(f"   Hedging Rate: {best_consistency['Advanced Hedging Rate']:.2%}")
        
        print(f"\n⚠️  Highest Variance: {worst_variance['Model']}")
        print(f"   Variance: {worst_variance['Variance']:.4f}")
        
        print(f"\n🤔 Most Hedging: {most_hedging['Model']}")
        print(f"   Hedging Rate: {most_hedging['Advanced Hedging Rate']:.2%}")
    
    # Statistical significance
    if statistical_analysis and 'ranking' in statistical_analysis:
        print(f"\n🏆 OVERALL RANKING:")
        for rank in statistical_analysis['ranking']['rankings']:
            print(f"  {rank['rank']}. {rank['model']} (Score: {rank['composite_score']:.3f})")
    
    return best_consistency, worst_variance, most_hedging


def export_reports(summary_df, best_consistency, worst_variance, most_hedging, statistical_analysis):
    """Export comprehensive reports"""
    print("\n" + "="*80)
    print("💾 EXPORTING REPORTS")
    print("="*80)
    
    # Save CSV
    csv_path = 'results/metrics/comprehensive_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"✅ CSV exported: {csv_path}")
    
    # Create detailed markdown report
    report = f"""# Comprehensive LLM Failure-Oriented Evaluation Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive evaluation of {len(summary_df)} language models across multiple failure modes, including contradictory contexts, response variance, and advanced reasoning scenarios.

## Summary Statistics

{summary_df.to_markdown(index=False)}

## Key Findings

### Best Performing Models

- **Most Consistent:** {best_consistency['Model']} (Consistency: {best_consistency.get('Advanced Consistency', best_consistency['Basic Consistency']):.3f})
- **Lowest Variance:** {summary_df.loc[summary_df['Variance'].idxmin()]['Model']} (Variance: {summary_df['Variance'].min():.4f})
- **Least Hedging:** {summary_df.loc[summary_df.get('Advanced Hedging Rate', summary_df['Basic Hedging Rate']).idxmin()]['Model']}

### Performance Patterns

1. **Consistency Across Test Types**: Models show varying performance between basic and advanced test cases
2. **Variance-Hedging Relationship**: Models with higher variance tend to exhibit more hedging behavior
3. **Category-Specific Strengths**: Different models excel in different reasoning categories

"""

    if statistical_analysis and 'ranking' in statistical_analysis:
        report += "\n## Overall Model Ranking\n\n"
        report += "| Rank | Model | Consistency | Hedging Rate | Variance | Composite Score |\n"
        report += "|------|-------|-------------|--------------|----------|----------------|\n"
        
        for rank in statistical_analysis['ranking']['rankings']:
            report += f"| {rank['rank']} | {rank['model']} | {rank['consistency']:.3f} | {rank['hedging_rate']:.2%} | {rank['variance']:.4f} | {rank['composite_score']:.3f} |\n"
    
    report += """

## Methodology

### Test Categories
- **Multi-hop Reasoning**: Chain of inference with contradictory steps
- **Temporal Reasoning**: Timeline conflicts and calculations
- **Numerical Reasoning**: Quantitative conflicts and precision
- **Scientific Facts**: Domain knowledge with subtle conflicts
- **Logical Contradictions**: Formal logic consistency
- **Historical Facts**: Factual accuracy with temporal elements

### Metrics
- **Consistency Score**: Semantic similarity between responses to conflicting contexts
- **Hedging Rate**: Frequency of uncertainty markers
- **Response Variance**: Stability across multiple runs
- **Statistical Significance**: T-tests and effect sizes for model comparisons

## Interpretation

Lower consistency scores indicate difficulty reconciling contradictions.
Higher variance suggests unstable behavior across runs.
Hedging behavior may indicate appropriate uncertainty or lack of confidence.

## Visualizations

See `results/visualizations/` for detailed charts:
- `comprehensive_comparison.png` - Overall model comparison
- `category_breakdown.png` - Performance by reasoning category
- `difficulty_analysis.png` - Performance by difficulty level
- `statistical_significance.png` - Statistical comparisons between models

---

*This evaluation framework focuses on failure modes rather than accuracy metrics, providing diagnostic insights for model selection and deployment decisions.*
"""
    
    report_path = 'results/COMPREHENSIVE_EVALUATION_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Comprehensive report: {report_path}")


def main():
    """Run complete analysis pipeline"""
    print("\n🚀 STARTING COMPREHENSIVE ANALYSIS PIPELINE")
    
    # Load data
    all_results, statistical_analysis = load_results()
    
    if not all_results:
        print("\n❌ No results found! Run evaluation first.")
        return
    
    # Create summary
    summary_df = create_summary_dataframe(all_results)
    
    # Generate visualizations
    plot_overall_comparison(summary_df)
    plot_category_breakdown(all_results)
    plot_difficulty_analysis(all_results)
    plot_statistical_significance(statistical_analysis)
    
    # Print findings
    best_consistency, worst_variance, most_hedging = print_key_findings(summary_df, statistical_analysis)
    
    # Export reports
    export_reports(summary_df, best_consistency, worst_variance, most_hedging, statistical_analysis)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/visualizations/comprehensive_comparison.png")
    print("  - results/visualizations/category_breakdown.png")
    print("  - results/visualizations/difficulty_analysis.png")
    print("  - results/visualizations/statistical_significance.png")
    print("  - results/metrics/comprehensive_summary.csv")
    print("  - results/COMPREHENSIVE_EVALUATION_REPORT.md")
    print("\n")


if __name__ == "__main__":
    main()