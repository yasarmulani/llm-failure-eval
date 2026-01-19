"""
Advanced Test Case Generator for Research-Grade LLM Evaluation
Designed for comprehensive failure mode analysis across multiple reasoning types
"""

from typing import List, Dict
import random


class AdvancedTestCaseGenerator:
    """
    Generate challenging test cases spanning multiple cognitive domains:
    - Multi-hop reasoning
    - Temporal reasoning  
    - Numerical conflicts
    - Logical contradictions
    - Scientific facts
    - Causal reasoning
    """
    
    def __init__(self):
        self.test_cases = []
        self._initialize_test_cases()
    
    def _initialize_test_cases(self):
        """Initialize comprehensive test case library"""
        
        # ============================================================
        # CATEGORY 1: Multi-Hop Reasoning (6 cases)
        # Tests: Chain of inference with conflicting intermediate steps
        # ============================================================
        self.test_cases.extend([
            {
                "category": "multi_hop_reasoning",
                "difficulty": "hard",
                "context_a": "The Amazon rainforest produces 20% of Earth's oxygen. Brazil contains 60% of the Amazon. Brazil's portion covers 5.5 million km².",
                "context_b": "The Amazon rainforest produces 6% of Earth's oxygen. Brazil contains 60% of the Amazon. Brazil's portion covers 5.5 million km².",
                "question": "What percentage of Earth's oxygen comes from Brazilian Amazon forests?",
                "ground_truth": "Context conflict: 12% vs 3.6%",
                "expected_conflict": True
            },
            {
                "category": "multi_hop_reasoning",
                "difficulty": "hard",
                "context_a": "Company X had $100M revenue in 2020. Revenue grew 50% in 2021, then grew 20% in 2022.",
                "context_b": "Company X had $100M revenue in 2020. Revenue grew 50% in 2021, then declined 20% in 2022.",
                "question": "What was Company X's total revenue growth from 2020 to 2022?",
                "ground_truth": "Context conflict: 80% growth vs 20% growth",
                "expected_conflict": True
            },
            {
                "category": "multi_hop_reasoning",
                "difficulty": "medium",
                "context_a": "Alice is 10cm taller than Bob. Bob is 5cm taller than Carol. Carol is 160cm tall.",
                "context_b": "Alice is 10cm taller than Bob. Carol is 3cm taller than Bob. Carol is 160cm tall.",
                "question": "Rank Alice, Bob, and Carol by height and state their exact heights.",
                "ground_truth": "Context conflict in ordering",
                "expected_conflict": True
            },
            {
                "category": "multi_hop_reasoning",
                "difficulty": "hard",
                "context_a": "Train A travels 100 km/h and leaves at 9 AM. Train B travels 120 km/h and leaves at 10 AM. Distance between cities is 600 km.",
                "context_b": "Train A travels 100 km/h and leaves at 9 AM. Train B travels 150 km/h and leaves at 10 AM. Distance between cities is 600 km.",
                "question": "When and where do the trains meet?",
                "ground_truth": "Context conflict: different meeting times",
                "expected_conflict": True
            },
            {
                "category": "multi_hop_reasoning",
                "difficulty": "medium",
                "context_a": "If all A are B, and all B are C, then all A are C. X is an A.",
                "context_b": "If all A are B, and some B are C, then all A are C. X is an A.",
                "question": "Can we definitively say X is a C?",
                "ground_truth": "Context conflict: logical validity",
                "expected_conflict": True
            },
            {
                "category": "multi_hop_reasoning",
                "difficulty": "hard",
                "context_a": "Country has 100M population. GDP is $5 trillion. 60% work in services, 30% in manufacturing, 10% in agriculture.",
                "context_b": "Country has 100M population. GDP is $5 trillion. 50% work in services, 30% in manufacturing, 20% in agriculture.",
                "question": "What is the GDP per capita and which sector employs the most people?",
                "ground_truth": "Context conflict: sector percentages don't match",
                "expected_conflict": True
            }
        ])
        
        # ============================================================
        # CATEGORY 2: Temporal Reasoning (5 cases)
        # Tests: Timeline conflicts and temporal calculations
        # ============================================================
        self.test_cases.extend([
            {
                "category": "temporal_reasoning",
                "difficulty": "medium",
                "context_a": "Treaty of Versailles signed June 28, 1919. WWI ended November 11, 1918. Negotiations took 6 months.",
                "context_b": "Treaty of Versailles signed June 28, 1920. WWI ended November 11, 1918. Negotiations took 6 months.",
                "question": "When did peace negotiations begin after WWI?",
                "ground_truth": "Context conflict: 1919 vs 1920 signing",
                "expected_conflict": True
            },
            {
                "category": "temporal_reasoning",
                "difficulty": "hard",
                "context_a": "Einstein published special relativity in 1905. He received Nobel Prize in 1921, 16 years later.",
                "context_b": "Einstein published special relativity in 1905. He received Nobel Prize in 1921, 20 years later.",
                "question": "How many years after special relativity did Einstein win the Nobel Prize?",
                "ground_truth": "Context conflict: 16 vs 20 years (1921-1905=16, so context_b is wrong)",
                "expected_conflict": True
            },
            {
                "category": "temporal_reasoning",
                "difficulty": "medium",
                "context_a": "Apollo 11 launched July 16, 1969. Moon landing occurred July 20, 1969. Return to Earth July 24, 1969.",
                "context_b": "Apollo 11 launched July 16, 1969. Moon landing occurred July 21, 1969. Return to Earth July 24, 1969.",
                "question": "How many days was the Apollo 11 mission from launch to landing on moon?",
                "ground_truth": "Context conflict: 4 days vs 5 days",
                "expected_conflict": True
            },
            {
                "category": "temporal_reasoning",
                "difficulty": "hard",
                "context_a": "iPhone released June 2007. iPhone 3G released July 2008, 13 months later. iPhone 4 released June 2010.",
                "context_b": "iPhone released June 2007. iPhone 3G released July 2008, 14 months later. iPhone 4 released June 2010.",
                "question": "What was the average time between iPhone releases from original to iPhone 4?",
                "ground_truth": "Context conflict in intermediate calculation",
                "expected_conflict": True
            },
            {
                "category": "temporal_reasoning",
                "difficulty": "medium",
                "context_a": "Declaration of Independence signed July 4, 1776. Constitution ratified June 21, 1788, 12 years later.",
                "context_b": "Declaration of Independence signed July 4, 1776. Constitution ratified June 21, 1788, 11 years later.",
                "question": "How long was the period between Declaration and Constitution?",
                "ground_truth": "Context conflict: calculation error (actual is ~12 years)",
                "expected_conflict": True
            }
        ])
        
        # ============================================================
        # CATEGORY 3: Numerical/Quantitative Conflicts (6 cases)
        # Tests: Numerical precision and calculation conflicts
        # ============================================================
        self.test_cases.extend([
            {
                "category": "numerical_reasoning",
                "difficulty": "easy",
                "context_a": "Mount Everest is 8,849m tall. K2 is 237m shorter than Everest.",
                "context_b": "Mount Everest is 8,849m tall. K2 is 611m shorter than Everest.",
                "question": "What is the exact height of K2 in meters?",
                "ground_truth": "Context conflict: 8,612m vs 8,238m",
                "expected_conflict": True
            },
            {
                "category": "numerical_reasoning",
                "difficulty": "hard",
                "context_a": "Speed of light is 299,792 km/s. Light takes 8.0 minutes to reach Earth from Sun.",
                "context_b": "Speed of light is 299,792 km/s. Light takes 8.3 minutes to reach Earth from Sun.",
                "question": "Calculate the distance from Sun to Earth in million kilometers.",
                "ground_truth": "Context conflict: 143.9M km vs 149.2M km",
                "expected_conflict": True
            },
            {
                "category": "numerical_reasoning",
                "difficulty": "medium",
                "context_a": "Marathon is 42.195 km. Elite runners finish in 2h 10min. Average runners take 4h 30min.",
                "context_b": "Marathon is 42.195 km. Elite runners finish in 2h 10min. Average runners take 4h 00min.",
                "question": "What is the pace difference (min/km) between elite and average runners?",
                "ground_truth": "Context conflict in average time",
                "expected_conflict": True
            },
            {
                "category": "numerical_reasoning",
                "difficulty": "hard",
                "context_a": "Compound interest: $10,000 invested at 7% annual return for 10 years grows to $19,672.",
                "context_b": "Compound interest: $10,000 invested at 7% annual return for 10 years grows to $20,000.",
                "question": "What is the correct final amount and total return percentage?",
                "ground_truth": "Context conflict: ~96.7% vs 100% return",
                "expected_conflict": True
            },
            {
                "category": "numerical_reasoning",
                "difficulty": "medium",
                "context_a": "Earth's circumference is 40,075 km at equator. Traveling at 1000 km/h takes 40 hours to circumnavigate.",
                "context_b": "Earth's circumference is 40,075 km at equator. Traveling at 1000 km/h takes 38 hours to circumnavigate.",
                "question": "How long does it take to fly around Earth at 1000 km/h?",
                "ground_truth": "Context conflict: 40h (correct) vs 38h",
                "expected_conflict": True
            },
            {
                "category": "numerical_reasoning",
                "difficulty": "hard",
                "context_a": "Population grows from 1M to 1.5M in 10 years. That's 50% growth, or 4.14% annual compound rate.",
                "context_b": "Population grows from 1M to 1.5M in 10 years. That's 50% growth, or 5% annual compound rate.",
                "question": "What is the correct annual compound growth rate?",
                "ground_truth": "Context conflict: 4.14% (correct) vs 5%",
                "expected_conflict": True
            }
        ])
        
        # ============================================================
        # CATEGORY 4: Scientific Facts (5 cases)
        # Tests: Domain knowledge with subtle factual conflicts
        # ============================================================
        self.test_cases.extend([
            {
                "category": "scientific_facts",
                "difficulty": "easy",
                "context_a": "Normal human body temperature is 37°C (98.6°F). Fever threshold is 38°C (100.4°F).",
                "context_b": "Normal human body temperature is 37°C (98.6°F). Fever threshold is 37.5°C (99.5°F).",
                "question": "At what temperature should a person be considered to have a fever?",
                "ground_truth": "Context conflict: clinical definitions vary",
                "expected_conflict": True
            },
            {
                "category": "scientific_facts",
                "difficulty": "medium",
                "context_a": "Earth's atmosphere: 78% nitrogen, 21% oxygen, 1% other gases (mostly argon).",
                "context_b": "Earth's atmosphere: 78% nitrogen, 20% oxygen, 2% other gases (mostly argon).",
                "question": "What is the exact oxygen percentage in Earth's atmosphere?",
                "ground_truth": "Context conflict: 21% (correct) vs 20%",
                "expected_conflict": True
            },
            {
                "category": "scientific_facts",
                "difficulty": "hard",
                "context_a": "Human brain weighs ~1.4 kg (2% of body weight) but uses 20% of body's energy/oxygen.",
                "context_b": "Human brain weighs ~1.4 kg (2% of body weight) but uses 25% of body's energy/oxygen.",
                "question": "What percentage of the body's energy does the brain consume?",
                "ground_truth": "Context conflict: 20% vs 25% (literature varies 20-25%)",
                "expected_conflict": True
            },
            {
                "category": "scientific_facts",
                "difficulty": "medium",
                "context_a": "DNA double helix discovered by Watson & Crick in 1953. Rosalind Franklin's X-ray data was crucial.",
                "context_b": "DNA double helix discovered by Watson & Crick in 1952. Rosalind Franklin's X-ray data was crucial.",
                "question": "In what year was the DNA double helix structure discovered?",
                "ground_truth": "Context conflict: 1953 (correct) vs 1952",
                "expected_conflict": True
            },
            {
                "category": "scientific_facts",
                "difficulty": "hard",
                "context_a": "Photosynthesis equation: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂. Efficiency is about 3-6% of sunlight.",
                "context_b": "Photosynthesis equation: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂. Efficiency is about 10-15% of sunlight.",
                "question": "What is the typical efficiency of photosynthesis in converting sunlight to chemical energy?",
                "ground_truth": "Context conflict: 3-6% (correct for overall) vs 10-15%",
                "expected_conflict": True
            }
        ])
        
        # ============================================================
        # CATEGORY 5: Logical Contradictions (4 cases)
        # Tests: Formal logic and consistency
        # ============================================================
        self.test_cases.extend([
            {
                "category": "logical_reasoning",
                "difficulty": "easy",
                "context_a": "All birds can fly. Penguins are birds. Penguins live in Antarctica.",
                "context_b": "Penguins cannot fly. Penguins are birds. Penguins live in Antarctica.",
                "question": "Based on the contexts, can penguins fly?",
                "ground_truth": "Logical contradiction between premises",
                "expected_conflict": True
            },
            {
                "category": "logical_reasoning",
                "difficulty": "medium",
                "context_a": "If it rains, the ground gets wet. It rained today. The ground is wet.",
                "context_b": "If it rains, the ground gets wet. It rained today. The ground is dry.",
                "question": "Is the ground wet or dry?",
                "ground_truth": "Logical contradiction: modus ponens violated",
                "expected_conflict": True
            },
            {
                "category": "logical_reasoning",
                "difficulty": "hard",
                "context_a": "All mammals are warm-blooded. Whales are mammals. Therefore whales are warm-blooded and have lungs.",
                "context_b": "All mammals are warm-blooded. Whales are mammals. Therefore whales are cold-blooded and have gills.",
                "question": "Are whales warm-blooded or cold-blooded, and do they have lungs or gills?",
                "ground_truth": "Logical contradiction in conclusion",
                "expected_conflict": True
            },
            {
                "category": "logical_reasoning",
                "difficulty": "medium",
                "context_a": "If P then Q. P is true. Therefore Q is true. Q implies R.",
                "context_b": "If P then Q. P is true. Therefore Q is false. Q implies R.",
                "question": "What can we conclude about R?",
                "ground_truth": "Logical contradiction in modus ponens",
                "expected_conflict": True
            }
        ])
        
        # ============================================================
        # CATEGORY 6: Historical Events (4 cases)
        # Tests: Factual accuracy with temporal elements
        # ============================================================
        self.test_cases.extend([
            {
                "category": "historical_facts",
                "difficulty": "medium",
                "context_a": "Christopher Columbus reached Americas in 1492 on his first voyage. He made 4 total voyages between 1492-1504.",
                "context_b": "Christopher Columbus reached Americas in 1492 on his first voyage. He made 3 total voyages between 1492-1504.",
                "question": "How many voyages did Columbus make to the Americas?",
                "ground_truth": "Context conflict: 4 (correct) vs 3",
                "expected_conflict": True
            },
            {
                "category": "historical_facts",
                "difficulty": "easy",
                "context_a": "Apollo 11 landed on Moon July 20, 1969. Neil Armstrong first stepped on Moon at 10:56 PM EDT.",
                "context_b": "Apollo 11 landed on Moon July 21, 1969. Neil Armstrong first stepped on Moon at 10:56 PM EDT.",
                "question": "On what date did humans first walk on the Moon?",
                "ground_truth": "Context conflict: July 20 (correct) vs July 21",
                "expected_conflict": True
            },
            {
                "category": "historical_facts",
                "difficulty": "hard",
                "context_a": "Berlin Wall built in 1961 during Cold War. It stood for 28 years before falling in November 1989.",
                "context_b": "Berlin Wall built in 1961 during Cold War. It stood for 30 years before falling in November 1989.",
                "question": "How long did the Berlin Wall stand?",
                "ground_truth": "Context conflict: 28 years (correct: 1961-1989) vs 30 years",
                "expected_conflict": True
            },
            {
                "category": "historical_facts",
                "difficulty": "medium",
                "context_a": "WWI: 1914-1918 (4 years). Estimated 17 million deaths total (military + civilian).",
                "context_b": "WWI: 1914-1918 (4 years). Estimated 20 million deaths total (military + civilian).",
                "question": "What was the total death toll of World War I?",
                "ground_truth": "Context conflict: estimates vary 17-20M",
                "expected_conflict": True
            }
        ])
    
    def get_all_cases(self) -> List[Dict]:
        """Return all test cases (30 total)"""
        return self.test_cases
    
    def get_cases_by_category(self, category: str) -> List[Dict]:
        """Return test cases filtered by category"""
        return [case for case in self.test_cases if case['category'] == category]
    
    def get_cases_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Return test cases filtered by difficulty"""
        return [case for case in self.test_cases if case.get('difficulty') == difficulty]
    
    def get_sample(self, n: int = 20, balanced: bool = True) -> List[Dict]:
        """
        Get a sample of n test cases
        
        Args:
            n: Number of cases to return
            balanced: If True, sample evenly across categories
        """
        if balanced:
            categories = set(case['category'] for case in self.test_cases)
            per_category = n // len(categories)
            sample = []
            
            for cat in categories:
                cat_cases = self.get_cases_by_category(cat)
                sample.extend(random.sample(cat_cases, min(per_category, len(cat_cases))))
            
            # Fill remaining slots if needed
            if len(sample) < n:
                remaining = [c for c in self.test_cases if c not in sample]
                sample.extend(random.sample(remaining, n - len(sample)))
            
            return sample[:n]
        else:
            return random.sample(self.test_cases, min(n, len(self.test_cases)))
    
    def get_statistics(self) -> Dict:
        """Return statistics about the test suite"""
        categories = {}
        difficulties = {}
        
        for case in self.test_cases:
            cat = case['category']
            categories[cat] = categories.get(cat, 0) + 1
            
            if 'difficulty' in case:
                diff = case['difficulty']
                difficulties[diff] = difficulties.get(diff, 0) + 1
        
        return {
            'total_cases': len(self.test_cases),
            'categories': categories,
            'difficulties': difficulties
        }