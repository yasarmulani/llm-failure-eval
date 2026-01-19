# Comprehensive LLM Failure-Oriented Evaluation Report

**Generated:** 2026-01-19 01:37:44

## Executive Summary

This report presents a comprehensive evaluation of 4 language models across multiple failure modes, including contradictory contexts, response variance, and advanced reasoning scenarios.

## Summary Statistics

| Model        |   Basic Consistency |   Basic Hedging Rate |    Variance |   Similarity |   Advanced Consistency |   Advanced Hedging Rate |   Total Test Cases |
|:-------------|--------------------:|---------------------:|------------:|-------------:|-----------------------:|------------------------:|-------------------:|
| qwen2.5:1.5b |            0.842577 |             0        | 6.76258e-05 |     0.993176 |               0.865151 |                0        |                 30 |
| gemma:2b     |            0.855078 |             0        | 0.00154657  |     0.939725 |               0.893192 |                0        |                 30 |
| phi3:mini    |            0.815672 |             0.333333 | 0.00407923  |     0.875515 |               0.786439 |                0.166667 |                 30 |
| llama3.2:1b  |            0.803078 |             0        | 0.00124851  |     0.876737 |               0.906626 |                0        |                 30 |

## Key Findings

### Best Performing Models

- **Most Consistent:** llama3.2:1b (Consistency: 0.907)
- **Lowest Variance:** qwen2.5:1.5b (Variance: 0.0001)
- **Least Hedging:** qwen2.5:1.5b

### Performance Patterns

1. **Consistency Across Test Types**: Models show varying performance between basic and advanced test cases
2. **Variance-Hedging Relationship**: Models with higher variance tend to exhibit more hedging behavior
3. **Category-Specific Strengths**: Different models excel in different reasoning categories


## Overall Model Ranking

| Rank | Model | Consistency | Hedging Rate | Variance | Composite Score |
|------|-------|-------------|--------------|----------|----------------|
| 1 | llama3.2:1b | 0.907 | 0.00% | 0.0012 | 0.894 |
| 2 | gemma:2b | 0.893 | 0.00% | 0.0015 | 0.878 |
| 3 | qwen2.5:1.5b | 0.865 | 0.00% | 0.0001 | 0.864 |
| 4 | phi3:mini | 0.786 | 16.67% | 0.0041 | 0.696 |


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
