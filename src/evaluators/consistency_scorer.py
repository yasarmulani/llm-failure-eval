import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ConsistencyScorer:
    """Measure response variance and consistency"""
    
    def __init__(self):
        # Load semantic similarity model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_variance(self, responses: List[str]) -> Dict[str, float]:
        """Calculate semantic variance across multiple responses"""
        if len(responses) < 2:
            return {"variance": 0.0, "mean_similarity": 1.0}
        
        # Get embeddings
        embeddings = self.model.encode(responses)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        
        variance = float(np.var(upper_triangle))
        mean_similarity = float(np.mean(upper_triangle))
        
        return {
            "variance": variance,
            "mean_similarity": mean_similarity,
            "std_similarity": float(np.std(upper_triangle)),
            "min_similarity": float(np.min(upper_triangle)),
            "max_similarity": float(np.max(upper_triangle))
        }
    
    def compare_responses(self, response_a: str, response_b: str) -> float:
        """Compare two responses semantically"""
        embeddings = self.model.encode([response_a, response_b])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def detect_flip_flop(self, responses: List[str], threshold: float = 0.5) -> bool:
        """Detect if model flip-flops between different answers"""
        if len(responses) < 2:
            return False
        
        embeddings = self.model.encode(responses)
        similarities = cosine_similarity(embeddings)
        
        # Check for low similarity pairs (potential flip-flops)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        flip_flops = np.sum(upper_triangle < threshold)
        
        return flip_flops > 0