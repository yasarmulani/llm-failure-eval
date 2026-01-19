import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np


def save_results(results: Dict, filename: str, results_dir: str = "results/metrics"):
    """Save evaluation results to JSON"""
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"{filename}_{timestamp}.json")
    
    # Convert numpy/bool types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict:
    """Load results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results list to pandas DataFrame"""
    return pd.DataFrame(results)


def print_summary(results: Dict):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("="*50 + "\n")