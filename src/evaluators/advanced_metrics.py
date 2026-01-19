"""
Advanced Metrics for Research-Grade Evaluation
Includes statistical significance, effect sizes, and calibration metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from collections import Counter


class AdvancedMetrics:
    """
    Advanced statistical and calibration metrics for LLM evaluation
    
    Metrics include:
    - Statistical significance tests (t-test, Mann-Whitney U)
    - Effect sizes (Cohen's d, Cliff's Delta)
    - Correlation analysis
    - Confidence calibration
    """
    
    def __init__(self):
        pass
    
    def compare_models_statistical(self, 
                                   model_a_scores: List[float], 
                                   model_b_scores: List[float],
                                   alpha: float = 0.05) -> Dict:
        """
        Compare two models using statistical tests
        
        Args:
            model_a_scores: Consistency scores for model A
            model_b_scores: Consistency scores for model B
            alpha: Significance level (default 0.05)
        
        Returns:
            Dictionary with test results and effect sizes
        """
        # T-test (parametric)
        t_stat, t_pvalue = stats.ttest_ind(model_a_scores, model_b_scores)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(model_a_scores, model_b_scores, alternative='two-sided')
        
        # Cohen's d (effect size)
        cohens_d = self._cohens_d(model_a_scores, model_b_scores)
        
        # Cliff's Delta (non-parametric effect size)
        cliffs_delta = self._cliffs_delta(model_a_scores, model_b_scores)
        
        # Interpretation
        t_significant = t_pvalue < alpha
        u_significant = u_pvalue < alpha
        
        effect_interpretation = self._interpret_effect_size(abs(cohens_d))
        
        return {
            "t_test": {
                "statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": bool(t_significant),
                "alpha": alpha
            },
            "mann_whitney_u": {
                "statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significant": bool(u_significant),
                "alpha": alpha
            },
            "effect_sizes": {
                "cohens_d": float(cohens_d),
                "cliffs_delta": float(cliffs_delta),
                "interpretation": effect_interpretation
            },
            "descriptive_stats": {
                "model_a_mean": float(np.mean(model_a_scores)),
                "model_a_std": float(np.std(model_a_scores)),
                "model_b_mean": float(np.mean(model_b_scores)),
                "model_b_std": float(np.std(model_b_scores)),
                "difference": float(np.mean(model_a_scores) - np.mean(model_b_scores))
            }
        }
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's Delta (non-parametric effect size)"""
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        dominance = 0
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1
                elif x < y:
                    dominance -= 1
        
        return dominance / (n1 * n2)
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def calculate_correlation(self, 
                            variance_scores: List[float], 
                            hedging_indicators: List[int]) -> Dict:
        """
        Calculate correlation between variance and hedging behavior
        
        Args:
            variance_scores: List of variance scores
            hedging_indicators: List of binary hedging indicators (0 or 1)
        
        Returns:
            Correlation statistics
        """
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(variance_scores, hedging_indicators)
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = stats.spearmanr(variance_scores, hedging_indicators)
        
        return {
            "pearson": {
                "correlation": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": bool(pearson_p < 0.05)
            },
            "spearman": {
                "correlation": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": bool(spearman_p < 0.05)
            }
        }
    
    def confidence_calibration(self, 
                              hedging_rates_by_difficulty: Dict[str, float]) -> Dict:
        """
        Analyze if model hedging is calibrated to difficulty
        
        Args:
            hedging_rates_by_difficulty: Dict mapping difficulty level to hedging rate
        
        Returns:
            Calibration analysis
        """
        difficulties = ['easy', 'medium', 'hard']
        rates = [hedging_rates_by_difficulty.get(d, 0) for d in difficulties]
        
        # Check if hedging increases with difficulty (expected behavior)
        is_monotonic = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
        
        # Calculate correlation between difficulty rank and hedging rate
        difficulty_ranks = list(range(len(difficulties)))
        if len(rates) > 1 and not all(r == rates[0] for r in rates):
            corr, p_value = stats.spearmanr(difficulty_ranks, rates)
        else:
            corr, p_value = 0.0, 1.0
        
        return {
            "is_calibrated": is_monotonic,
            "difficulty_hedging_correlation": float(corr),
            "p_value": float(p_value),
            "rates_by_difficulty": hedging_rates_by_difficulty,
            "interpretation": "well-calibrated" if is_monotonic else "poorly-calibrated"
        }
    
    def bootstrap_confidence_interval(self, 
                                     scores: List[float], 
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for mean scores
        
        Args:
            scores: List of scores
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)
        
        Returns:
            Confidence interval statistics
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            "mean": float(np.mean(scores)),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": confidence_level
            },
            "bootstrap_std": float(np.std(bootstrap_means)),
            "n_bootstrap": n_bootstrap
        }
    
    def category_performance_analysis(self, results_by_category: Dict[str, List[float]]) -> Dict:
        """
        Analyze performance differences across categories
        
        Args:
            results_by_category: Dict mapping category name to list of consistency scores
        
        Returns:
            ANOVA results and post-hoc comparisons
        """
        categories = list(results_by_category.keys())
        scores = list(results_by_category.values())
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*scores)
        
        # Pairwise comparisons (if significant)
        pairwise_comparisons = {}
        if p_value < 0.05:
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    cat_a, cat_b = categories[i], categories[j]
                    t_stat, t_p = stats.ttest_ind(scores[i], scores[j])
                    
                    pairwise_comparisons[f"{cat_a}_vs_{cat_b}"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(t_p),
                        "significant": bool(t_p < 0.05),
                        "mean_difference": float(np.mean(scores[i]) - np.mean(scores[j]))
                    }
        
        return {
            "anova": {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            },
            "pairwise_comparisons": pairwise_comparisons,
            "category_means": {cat: float(np.mean(scores[i])) for i, cat in enumerate(categories)}
        }