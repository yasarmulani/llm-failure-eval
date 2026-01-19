import sys
sys.path.append('..')

from src.generators.contradictory_context import ContradictoryContextGenerator
from src.evaluators.consistency_scorer import ConsistencyScorer
from src.evaluators.contradiction_detector import ContradictionDetector


def test_contradictory_generator():
    """Test contradictory context generator"""
    generator = ContradictoryContextGenerator()
    prompts = generator.generate_prompts()
    
    assert len(prompts) > 0, "Should generate prompts"
    assert 'question' in prompts[0], "Should have question"
    assert 'prompt_contradictory' in prompts[0], "Should have contradictory prompt"
    print("✅ test_contradictory_generator passed")


def test_consistency_scorer():
    """Test consistency scorer"""
    scorer = ConsistencyScorer()
    
    # Test identical responses
    responses = ["The sky is blue"] * 3
    metrics = scorer.calculate_variance(responses)
    
    assert metrics['mean_similarity'] > 0.99, "Identical responses should have high similarity"
    print("✅ test_consistency_scorer passed")


def test_contradiction_detector():
    """Test contradiction detector"""
    detector = ContradictionDetector()
    
    # Test hedging detection
    response = "I'm not sure, but maybe it could be in Paris"
    result = detector.detect_hedging(response)
    
    assert result['has_hedging'], "Should detect hedging"
    print("✅ test_contradiction_detector passed")


if __name__ == "__main__":
    test_contradictory_generator()
    test_consistency_scorer()
    test_contradiction_detector()
    print("\n✅ All tests passed!")