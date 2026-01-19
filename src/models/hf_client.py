import requests
import time
from typing import List


class OllamaClient:
    """Wrapper for Ollama local API"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 256) -> str:
        """Generate response from local Ollama model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            print(f"Error with {self.model_name}: {e}")
            return ""
    
    def generate_multiple(self, prompt: str, n: int = 5, 
                         temperature: float = 0.7) -> List[str]:
        """Generate multiple responses"""
        responses = []
        for i in range(n):
            response = self.generate(prompt, temperature=temperature)
            responses.append(response)
            time.sleep(0.1)
        return responses
    
    def __repr__(self):
        return f"OllamaClient(model={self.model_name})"