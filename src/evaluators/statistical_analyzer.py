"""
Statistical Analysis Module for Multi-Model Comparison
Performs comprehensive statistical analysis across all evaluated models
"""

from typing import Dict, List
import numpy as np
from src.evaluators.advanced_metrics import AdvancedMetrics


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis across multiple models
    
    Provides:
    - Cross-model comparisons
    - Statistical significance testing
    - Effect size calculations
    - Category-wise performance analysis
    """
    
    def __init__(self):
        self.metrics = AdvancedMetrics()
    
    def analyze_all_models(self, all_results: List[Dict]) -> Dict:
        """
        Perform comprehensive statistical analysis across all models
        
        Args:
            all_results: List of result dictionaries from each model
        
        Returns:
            Complete statistical analysis
        """
        analysis = {
            "models_compared": [r['model'] for r in all_results],
            "pairwise_comparisons": {},
            "ranking": {},
            "category_analysis": {},
            "difficulty_analysis": {},
            "correlation_analysis": {}
        }
        
        # Extract model names
        models = [r['model'] for r in all_results]
        
        # Pairwise statistical comparisons
        analysis['pairwise_comparisons'] = self._pairwise_comparisons(all_results)
        
        # Overall ranking
        analysis['ranking'] = self._rank_models(all_results)
        
        # Category-wise analysis
        if any('advanced_cases' in r and r['advanced_cases'] for r in all_results):
            analysis['category_analysis'] = self._category_analysis(all_results)
            analysis['difficulty_analysis'] = self._difficulty_analysis(all_results)
        
        # Correlation analysis (variance vs hedging)
        analysis['correlation_analysis'] = self._correlation_analysis(all_results)
        
        return analysis
    
    def _pairwise_comparisons(self, all_results: List[Dict]) -> Dict:
        """Compare each pair of models statistically"""
        comparisons = {}
        
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                model_a = all_results[i]['model']
                model_b = all_results[j]['model']
                
                # Extract consistency scores from advanced cases
                if 'advanced_cases' in all_results[i] and all_results[i]['advanced_cases']:
                    scores_a = [case['consistency_a_vs_b'] for case in all_results[i]['advanced_cases']]
                    scores_b = [case['consistency_a_vs_b'] for case in all_results[j]['advanced_cases']]
                    
                    comparison = self.metrics.compare_models_statistical(scores_a, scores_b)
                    
                    comparison_key = f"{model_a}_vs_{model_b}"
                    comparisons[comparison_key] = comparison
        
        return comparisons
    
    def _rank_models(self, all_results: List[Dict]) -> Dict:
        """Rank models by overall consistency"""
        rankings = []
        
        for result in all_results:
            model = result['model']
            
            # Get overall consistency from advanced evaluation
            if 'advanced_evaluation' in result['summary']:
                consistency = result['summary']['advanced_evaluation']['overall_consistency']
            else:
                consistency = result['summary']['contradictory_contexts']['avg_consistency_opposing_contexts']
            
            # Get hedging rate
            if 'advanced_evaluation' in result['summary']:
                hedging_rate = result['summary']['advanced_evaluation']['overall_hedging_rate']
            else:
                hedging_rate = result['summary']['contradictory_contexts']['hedging_rate']
            
            # Get variance
            variance = result['summary']['response_variance']['avg_variance']
            
            # Composite score (higher consistency, lower hedging, lower variance is better)
            composite_score = consistency - (0.3 * hedging_rate) - (10 * variance)
            
            rankings.append({
                "model": model,
                "consistency": consistency,
                "hedging_rate": hedging_rate,
                "variance": variance,
                "composite_score": composite_score
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return {
            "rankings": rankings,
            "best_model": rankings[0]['model'] if rankings else None,
            "worst_model": rankings[-1]['model'] if rankings else None
        }
    
    def _category_analysis(self, all_results: List[Dict]) -> Dict:
        """Analyze performance by category across models"""
        category_performance = {}
        
        # Collect scores by category for each model
        for result in all_results:
            model = result['model']
            
            if 'advanced_cases' not in result or not result['advanced_cases']:
                continue
            
            for case in result['advanced_cases']:
                category = case['category']
                
                if category not in category_performance:
                    category_performance[category] = {}
                
                if model not in category_performance[category]:
                    category_performance[category][model] = []
                
                category_performance[category][model].append(case['consistency_a_vs_b'])
        
        # Compute statistics for each category
        category_stats = {}
        for category, model_scores in category_performance.items():
            category_stats[category] = {
                "model_means": {model: float(np.mean(scores)) for model, scores in model_scores.items()},
                "best_model": max(model_scores.items(), key=lambda x: np.mean(x[1]))[0],
                "worst_model": min(model_scores.items(), key=lambda x: np.mean(x[1]))[0]
            }
            
            # ANOVA if multiple models
            if len(model_scores) > 1:
                scores_list = list(model_scores.values())
                anova_result = self.metrics.category_performance_analysis(model_scores)
                category_stats[category]['anova'] = anova_result
        
        return category_stats
    
    def _difficulty_analysis(self, all_results: List[Dict]) -> Dict:
        """Analyze performance by difficulty level"""
        difficulty_performance = {}
        
        for result in all_results:
            model = result['model']
            
            if 'advanced_cases' not in result or not result['advanced_cases']:
                continue
            
            for case in result['advanced_cases']:
                difficulty = case.get('difficulty', 'medium')
                
                if difficulty not in difficulty_performance:
                    difficulty_performance[difficulty] = {}
                
                if model not in difficulty_performance[difficulty]:
                    difficulty_performance[difficulty][model] = []
                
                difficulty_performance[difficulty][model].append(case['consistency_a_vs_b'])
        
        # Compute statistics
        difficulty_stats = {}
        for difficulty, model_scores in difficulty_performance.items():
            difficulty_stats[difficulty] = {
                "model_means": {model: float(np.mean(scores)) for model, scores in model_scores.items()},
                "overall_mean": float(np.mean([score for scores in model_scores.values() for score in scores]))
            }
        
        return difficulty_stats
    
    def _correlation_analysis(self, all_results: List[Dict]) -> Dict:
        """Analyze correlation between variance and hedging"""
        correlations = {}
        
        for result in all_results:
            model = result['model']
            
            if 'advanced_cases' not in result or not result['advanced_cases']:
                continue
            
            variances = []
            hedging_indicators = []
            
            # For each advanced case, we approximate variance from consistency
            for case in result['advanced_cases']:
                # Lower consistency = higher variance (approximate)
                variance_proxy = 1 - case['consistency_a_vs_b']
                variances.append(variance_proxy)
                
                hedging = 1 if case['contradiction_analysis']['hedging']['has_hedging'] else 0
                hedging_indicators.append(hedging)
            
            if len(variances) > 2:  # Need at least 3 points for correlation
                correlation = self.metrics.calculate_correlation(variances, hedging_indicators)
                correlations[model] = correlation
        
        return correlations