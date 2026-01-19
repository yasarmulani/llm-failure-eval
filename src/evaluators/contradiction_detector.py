from typing import List, Dict
import re


class ContradictionDetector:
    """Detect contradictions in model responses"""
    
    def __init__(self):
        # Keywords indicating uncertainty or hedging
        self.uncertainty_words = [
            "maybe", "perhaps", "possibly", "might", "could be",
            "unclear", "uncertain", "not sure", "conflicting"
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r"not .{1,30}",
            r"n't .{1,30}",
            r"neither .{1,30} nor",
            r"however",
            r"but "
        ]
    
    def detect_hedging(self, response: str) -> Dict[str, any]:
        """Detect if model is hedging due to contradictory context"""
        response_lower = response.lower()
        
        hedging_count = sum(1 for word in self.uncertainty_words 
                           if word in response_lower)
        
        has_hedging = hedging_count > 0
        
        return {
            "has_hedging": has_hedging,
            "hedging_count": hedging_count,
            "hedging_words_found": [w for w in self.uncertainty_words 
                                   if w in response_lower]
        }
    
    def detect_internal_contradiction(self, response: str) -> Dict[str, any]:
        """Detect contradictions within a single response"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {"has_contradiction": False, "confidence": 0.0}
        
        # Look for negation patterns
        negation_count = 0
        for pattern in self.negation_patterns:
            negation_count += len(re.findall(pattern, response.lower()))
        
        # Simple heuristic: multiple negations or hedging suggests contradiction
        has_contradiction = negation_count >= 2
        
        return {
            "has_contradiction": has_contradiction,
            "negation_count": negation_count,
            "sentence_count": len(sentences),
            "confidence": min(negation_count / len(sentences), 1.0)
        }
    
    def analyze_contradictory_response(self, response_a: str, 
                                      response_b: str, 
                                      response_both: str) -> Dict:
        """Analyze how model handles contradictory contexts"""
        
        hedge_analysis = self.detect_hedging(response_both)
        contradiction_analysis = self.detect_internal_contradiction(response_both)
        
        # Check if response changes significantly with contradictory context
        response_length_change = abs(len(response_both) - 
                                    (len(response_a) + len(response_b)) / 2)
        
        return {
            "hedging": hedge_analysis,
            "internal_contradiction": contradiction_analysis,
            "length_change": response_length_change,
            "response_a_length": len(response_a),
            "response_b_length": len(response_b),
            "response_both_length": len(response_both)
        }