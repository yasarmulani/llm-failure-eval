from typing import List, Dict
import random


class ContradictoryContextGenerator:
    """Generate test cases with contradictory information"""
    
    def __init__(self):
        self.templates = [
            {
                "context_a": "The Eiffel Tower is located in Paris, France.",
                "context_b": "The Eiffel Tower is located in London, England.",
                "question": "Where is the Eiffel Tower located?",
                "expected_conflict": True
            },
            {
                "context_a": "Python was created by Guido van Rossum in 1991.",
                "context_b": "Python was created by James Gosling in 1995.",
                "question": "Who created Python and when?",
                "expected_conflict": True
            },
            {
                "context_a": "Water boils at 100°C at sea level.",
                "context_b": "Water boils at 90°C at sea level.",
                "question": "At what temperature does water boil at sea level?",
                "expected_conflict": True
            }
        ]
    
    def generate_prompts(self, num_samples: int = None) -> List[Dict]:
        """Generate contradictory context prompts"""
        if num_samples:
            samples = random.sample(self.templates, min(num_samples, len(self.templates)))
        else:
            samples = self.templates
            
        prompts = []
        for sample in samples:
            # Version 1: Context A only
            prompt_a = f"Context: {sample['context_a']}\n\nQuestion: {sample['question']}\nAnswer:"
            
            # Version 2: Context B only  
            prompt_b = f"Context: {sample['context_b']}\n\nQuestion: {sample['question']}\nAnswer:"
            
            # Version 3: Both contexts (contradictory)
            prompt_both = f"Context: {sample['context_a']} {sample['context_b']}\n\nQuestion: {sample['question']}\nAnswer:"
            
            prompts.append({
                "question": sample['question'],
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "prompt_contradictory": prompt_both,
                "has_conflict": sample['expected_conflict']
            })
        
        return prompts
    
    def add_custom_case(self, context_a: str, context_b: str, 
                       question: str, has_conflict: bool = True):
        """Add custom contradictory case"""
        self.templates.append({
            "context_a": context_a,
            "context_b": context_b,
            "question": question,
            "expected_conflict": has_conflict
        })