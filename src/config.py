import os

# Hugging Face Token (get from https://huggingface.co/settings/tokens)
HF_TOKEN = os.getenv("HF_TOKEN", "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")


# Models to evaluate - add more for comprehensive comparison
MODELS = [
    "llama3.2:1b",
    "phi3:mini",
    "gemma:2b",       # Add this
    "qwen2.5:1.5b",   # Add this if available
]

# Evaluation settings
NUM_VARIANCE_RUNS = 5  # Test same prompt N times
TEMPERATURE = 0.7
MAX_TOKENS = 256

# Advanced evaluation settings
NUM_ADVANCED_CASES = 30  # Number of advanced test cases to run
RUN_ADVANCED_EVAL = True  # Enable/disable advanced evaluation

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"