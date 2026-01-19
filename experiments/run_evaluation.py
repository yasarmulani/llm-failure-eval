"""
Research-Grade LLM Failure Evaluation Pipeline
Comprehensive evaluation across multiple failure modes and reasoning categories
"""

import sys
sys.path.append('..')

from src.models.hf_client import OllamaClient as HFModelClient
from src.generators.contradictory_context import ContradictoryContextGenerator
from src.generators.advanced_test_cases import AdvancedTestCaseGenerator
from src.evaluators.consistency_scorer import ConsistencyScorer
from src.evaluators.statistical_analyzer import StatisticalAnalyzer
from src.evaluators.contradiction_detector import ContradictionDetector
from src.utils import save_results, print_summary
from src.config import MODELS, NUM_VARIANCE_RUNS
import time
from typing import Dict, List


class FailureEvaluator:
    """
    Main evaluation pipeline for LLM failure mode analysis
    
    Evaluates models on:
    1. Basic contradictory contexts (3 cases)
    2. Response variance (consistency across runs)
    3. Advanced test suite (30 cases across 6 categories)
    """
    
    def __init__(self, model_name: str):
        self.model = HFModelClient(model_name)
        self.model_name = model_name
        self.generator = ContradictoryContextGenerator()
        self.advanced_generator = AdvancedTestCaseGenerator()
        self.consistency_scorer = ConsistencyScorer()
        self.contradiction_detector = ContradictionDetector()
        
    def evaluate_contradictory_contexts(self) -> List[Dict]:
        """Evaluate model on basic contradictory context scenarios"""
        print(f"\n🔍 Evaluating {self.model_name} on contradictory contexts...")
        
        test_cases = self.generator.generate_prompts()
        results = []
        
        for idx, case in enumerate(test_cases):
            print(f"\n  Test case {idx+1}/{len(test_cases)}: {case['question']}")
            
            # Get responses for each context version
            response_a = self.model.generate(case['prompt_a'])
            time.sleep(0.5)
            
            response_b = self.model.generate(case['prompt_b'])
            time.sleep(0.5)
            
            response_both = self.model.generate(case['prompt_contradictory'])
            time.sleep(0.5)
            
            # Analyze responses
            contradiction_analysis = self.contradiction_detector.analyze_contradictory_response(
                response_a, response_b, response_both
            )
            
            # Compare consistency
            consistency_ab = self.consistency_scorer.compare_responses(response_a, response_b)
            consistency_a_both = self.consistency_scorer.compare_responses(response_a, response_both)
            consistency_b_both = self.consistency_scorer.compare_responses(response_b, response_both)
            
            result = {
                "question": case['question'],
                "response_a": response_a,
                "response_b": response_b,
                "response_contradictory": response_both,
                "consistency_a_vs_b": consistency_ab,
                "consistency_a_vs_both": consistency_a_both,
                "consistency_b_vs_both": consistency_b_both,
                "contradiction_analysis": contradiction_analysis,
                "has_conflict": case['has_conflict']
            }
            
            results.append(result)
            
            print(f"    Consistency (A vs B): {consistency_ab:.3f}")
            print(f"    Hedging detected: {contradiction_analysis['hedging']['has_hedging']}")
        
        return results
    
    def evaluate_response_variance(self, num_samples: int = 3) -> List[Dict]:
        """Evaluate response variance on same prompts"""
        print(f"\n🔍 Evaluating {self.model_name} on response variance...")
        
        test_cases = self.generator.generate_prompts(num_samples=num_samples)
        results = []
        
        for idx, case in enumerate(test_cases):
            print(f"\n  Test case {idx+1}/{len(test_cases)}: {case['question']}")
            
            # Generate multiple responses for same prompt
            responses = self.model.generate_multiple(
                case['prompt_a'], 
                n=NUM_VARIANCE_RUNS
            )
            
            # Calculate variance metrics
            variance_metrics = self.consistency_scorer.calculate_variance(responses)
            flip_flop = self.consistency_scorer.detect_flip_flop(responses)
            
            result = {
                "question": case['question'],
                "responses": responses,
                "variance_metrics": variance_metrics,
                "has_flip_flop": flip_flop,
                "num_runs": NUM_VARIANCE_RUNS
            }
            
            results.append(result)
            
            print(f"    Mean similarity: {variance_metrics['mean_similarity']:.3f}")
            print(f"    Variance: {variance_metrics['variance']:.4f}")
            print(f"    Flip-flop detected: {flip_flop}")
        
        return results
    
    def evaluate_advanced_cases(self, num_samples: int = 30) -> List[Dict]:
        """
        Evaluate on research-grade advanced test cases
        
        Test categories:
        - Multi-hop reasoning
        - Temporal reasoning
        - Numerical conflicts
        - Scientific facts
        - Logical contradictions
        - Historical facts
        """
        print(f"\n🔬 Evaluating {self.model_name} on advanced test suite...")
        print(f"   Test cases: {num_samples} (balanced across categories)")
        
        test_cases = self.advanced_generator.get_sample(num_samples, balanced=True)
        results = []
        
        # Show statistics
        stats = self.advanced_generator.get_statistics()
        print(f"   Total test library: {stats['total_cases']} cases")
        print(f"   Categories: {list(stats['categories'].keys())}")
        
        for idx, case in enumerate(test_cases):
            category = case['category'].replace('_', ' ').title()
            difficulty = case.get('difficulty', 'medium').upper()
            
            print(f"\n  [{idx+1}/{len(test_cases)}] {category} ({difficulty})")
            print(f"  Q: {case['question'][:70]}...")
            
            # Create prompts
            prompt_a = f"Context: {case['context_a']}\n\nQuestion: {case['question']}\nAnswer:"
            prompt_b = f"Context: {case['context_b']}\n\nQuestion: {case['question']}\nAnswer:"
            prompt_both = f"Context: {case['context_a']} {case['context_b']}\n\nQuestion: {case['question']}\nAnswer:"
            
            # Get responses
            response_a = self.model.generate(prompt_a)
            time.sleep(0.5)
            
            response_b = self.model.generate(prompt_b)
            time.sleep(0.5)
            
            response_both = self.model.generate(prompt_both)
            time.sleep(0.5)
            
            # Analyze
            contradiction_analysis = self.contradiction_detector.analyze_contradictory_response(
                response_a, response_b, response_both
            )
            
            consistency_ab = self.consistency_scorer.compare_responses(response_a, response_b)
            consistency_a_both = self.consistency_scorer.compare_responses(response_a, response_both)
            consistency_b_both = self.consistency_scorer.compare_responses(response_b, response_both)
            
            result = {
                "category": case['category'],
                "difficulty": case.get('difficulty', 'medium'),
                "question": case['question'],
                "ground_truth": case.get('ground_truth', 'N/A'),
                "response_a": response_a,
                "response_b": response_b,
                "response_contradictory": response_both,
                "consistency_a_vs_b": consistency_ab,
                "consistency_a_vs_both": consistency_a_both,
                "consistency_b_vs_both": consistency_b_both,
                "contradiction_analysis": contradiction_analysis,
                "has_conflict": case['expected_conflict']
            }
            
            results.append(result)
            
            # Show metrics
            print(f"     Consistency (A↔B): {consistency_ab:.3f}")
            print(f"     Hedging: {contradiction_analysis['hedging']['has_hedging']}")
            if contradiction_analysis['hedging']['has_hedging']:
                words = contradiction_analysis['hedging']['hedging_words_found'][:3]
                print(f"     → Words: {words}")
        
        return results
    
    def run_full_evaluation(self, include_advanced: bool = True) -> Dict:
        """
        Run complete evaluation suite
        
        Args:
            include_advanced: Whether to run advanced test cases (default: True)
        
        Returns:
            Dictionary containing all results and summary statistics
        """
        print(f"\n{'='*70}")
        print(f"🚀 STARTING COMPREHENSIVE EVALUATION: {self.model_name}")
        print(f"{'='*70}")
        
        # Run basic evaluations
        contradictory_results = self.evaluate_contradictory_contexts()
        variance_results = self.evaluate_response_variance()
        
        # Run advanced evaluation
        advanced_results = []
        if include_advanced:
            advanced_results = self.evaluate_advanced_cases(num_samples=30)
        
        # Aggregate metrics
        summary = self._compute_summary(
            contradictory_results, 
            variance_results,
            advanced_results
        )
        
        full_results = {
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "contradictory_contexts": contradictory_results,
            "response_variance": variance_results,
            "advanced_cases": advanced_results
        }
        
        # Save results
        filename = self.model_name.replace("/", "_").replace(":", "_")
        save_results(full_results, filename)
        
        # Print summary
        print_summary(summary)
        
        return full_results
    
    def _compute_summary(self, 
                        contradictory_results: List[Dict], 
                        variance_results: List[Dict],
                        advanced_results: List[Dict] = None) -> Dict:
        """
        Compute comprehensive aggregate statistics
        
        Includes:
        - Basic contradictory context metrics
        - Response variance metrics
        - Advanced test suite breakdown by category
        """
        
        # Basic contradictory context metrics
        avg_consistency = sum(r['consistency_a_vs_b'] for r in contradictory_results) / len(contradictory_results)
        hedging_rate = sum(1 for r in contradictory_results 
                          if r['contradiction_analysis']['hedging']['has_hedging']) / len(contradictory_results)
        
        # Variance metrics
        avg_variance = sum(r['variance_metrics']['variance'] for r in variance_results) / len(variance_results)
        avg_similarity = sum(r['variance_metrics']['mean_similarity'] for r in variance_results) / len(variance_results)
        flip_flop_rate = sum(1 for r in variance_results if r['has_flip_flop']) / len(variance_results)
        
        summary = {
            "contradictory_contexts": {
                "avg_consistency_opposing_contexts": avg_consistency,
                "hedging_rate": hedging_rate,
                "total_cases": len(contradictory_results)
            },
            "response_variance": {
                "avg_variance": avg_variance,
                "avg_mean_similarity": avg_similarity,
                "flip_flop_rate": flip_flop_rate,
                "total_cases": len(variance_results),
                "runs_per_case": NUM_VARIANCE_RUNS
            }
        }
        
        # Advanced metrics
        if advanced_results and len(advanced_results) > 0:
            # Overall advanced metrics
            adv_avg_consistency = sum(r['consistency_a_vs_b'] for r in advanced_results) / len(advanced_results)
            adv_hedging_rate = sum(1 for r in advanced_results 
                                  if r['contradiction_analysis']['hedging']['has_hedging']) / len(advanced_results)
            
            # Per-category breakdown
            category_stats = {}
            for result in advanced_results:
                cat = result['category']
                if cat not in category_stats:
                    category_stats[cat] = {
                        'consistency_scores': [],
                        'hedging_count': 0,
                        'total': 0
                    }
                
                category_stats[cat]['consistency_scores'].append(result['consistency_a_vs_b'])
                if result['contradiction_analysis']['hedging']['has_hedging']:
                    category_stats[cat]['hedging_count'] += 1
                category_stats[cat]['total'] += 1
            
            # Compute per-category averages
            category_breakdown = {}
            for cat, stats in category_stats.items():
                category_breakdown[cat] = {
                    'avg_consistency': sum(stats['consistency_scores']) / len(stats['consistency_scores']),
                    'hedging_rate': stats['hedging_count'] / stats['total'],
                    'count': stats['total']
                }
            
            # Difficulty breakdown
            difficulty_stats = {}
            for result in advanced_results:
                diff = result.get('difficulty', 'medium')
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {
                        'consistency_scores': [],
                        'hedging_count': 0,
                        'total': 0
                    }
                
                difficulty_stats[diff]['consistency_scores'].append(result['consistency_a_vs_b'])
                if result['contradiction_analysis']['hedging']['has_hedging']:
                    difficulty_stats[diff]['hedging_count'] += 1
                difficulty_stats[diff]['total'] += 1
            
            difficulty_breakdown = {}
            for diff, stats in difficulty_stats.items():
                difficulty_breakdown[diff] = {
                    'avg_consistency': sum(stats['consistency_scores']) / len(stats['consistency_scores']),
                    'hedging_rate': stats['hedging_count'] / stats['total'],
                    'count': stats['total']
                }
            
            summary['advanced_evaluation'] = {
                'overall_consistency': adv_avg_consistency,
                'overall_hedging_rate': adv_hedging_rate,
                'total_cases': len(advanced_results),
                'category_breakdown': category_breakdown,
                'difficulty_breakdown': difficulty_breakdown
            }
        
        return summary


def main():
    """
    Run evaluation on all configured models with statistical analysis
    
    Models are configured in src/config.py
    Results are saved to results/metrics/
    """
    print("\n" + "="*70)
    print("🔬 RESEARCH-GRADE LLM FAILURE EVALUATION SUITE")
    print("="*70)
    print("\nEvaluating models:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")
    print()
    
    all_results = []
    
    # Run evaluation for each model
    for model_name in MODELS:
        try:
            evaluator = FailureEvaluator(model_name)
            results = evaluator.run_full_evaluation(include_advanced=True)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Perform cross-model statistical analysis
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("📊 PERFORMING STATISTICAL ANALYSIS")
        print(f"{'='*70}\n")
        
        analyzer = StatisticalAnalyzer()
        statistical_analysis = analyzer.analyze_all_models(all_results)
        
        # Save statistical analysis
        save_results(statistical_analysis, "statistical_analysis")
        
        # Print key findings
        print("\n🏆 MODEL RANKINGS:")
        for ranking in statistical_analysis['ranking']['rankings']:
            print(f"  {ranking['rank']}. {ranking['model']}")
            print(f"     Consistency: {ranking['consistency']:.3f}")
            print(f"     Hedging Rate: {ranking['hedging_rate']:.2%}")
            print(f"     Composite Score: {ranking['composite_score']:.3f}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: results/metrics/")
    print(f"Total models evaluated: {len(all_results)}/{len(MODELS)}")
    print()
    
    return all_results


if __name__ == "__main__":
    main()