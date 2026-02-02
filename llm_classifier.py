"""
LLM-based Taxonomy Classifier using Ollama + Llama 3

This module provides a locally-run LLM approach for procurement taxonomy classification.
Uses Ollama with Llama 3 for zero-shot or few-shot classification.

Prerequisites:
- Install Ollama: brew install ollama (macOS)
- Pull model: ollama pull llama3:8b
- Start service: ollama serve
"""

import json
import pandas as pd
from typing import Dict, List, Optional
import time

try:
    import ollama
except ImportError:
    print("Warning: ollama package not installed. Run: pip install ollama")
    ollama = None


class LLMTaxonomyClassifier:
    """
    LLM-based hierarchical taxonomy classifier using Ollama.

    Features:
    - Zero-shot classification (no training needed)
    - Few-shot learning (provide examples)
    - Explainable predictions
    - Handles Greek text naturally
    """

    def __init__(self, model_name='llama3:8b', temperature=0.1):
        """
        Initialize LLM classifier.

        Parameters:
        -----------
        model_name : str
            Ollama model to use (default: llama3:8b)
        temperature : float
            Sampling temperature (lower = more deterministic)
        """
        if ollama is None:
            raise ImportError("Ollama package not installed. Run: pip install ollama")

        self.model_name = model_name
        self.temperature = temperature
        self.taxonomy_hierarchy = None
        self.few_shot_examples = []

        # Test connection
        try:
            ollama.list()
            print(f"✓ Connected to Ollama")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama. Is the service running? Error: {e}")

    def load_taxonomy(self, excel_path, sheet_name='L1-L2-L3'):
        """
        Load official taxonomy hierarchy from Excel.

        Parameters:
        -----------
        excel_path : str
            Path to Excel file with taxonomy
        sheet_name : str
            Sheet name containing taxonomy
        """
        print(f"\nLoading taxonomy from '{sheet_name}' sheet...")

        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Detect columns
        l1_col = 'L1' if 'L1' in df.columns else 'Επίπεδο Κατηγοριοποίησης L.1'
        l2_col = 'L2' if 'L2' in df.columns else 'Επίπεδο Κατηγοριοποίησης L.2'
        l3_col = 'L3' if 'L3' in df.columns else 'Επίπεδο Κατηγοριοποίησης L.3'

        # Forward fill merged cells
        df[l1_col] = df[l1_col].ffill()
        df[l2_col] = df[l2_col].ffill()
        df = df.dropna(subset=[l3_col])

        # Build hierarchy
        hierarchy = {}
        for _, row in df.iterrows():
            l1 = row[l1_col]
            l2 = row[l2_col]
            l3 = row[l3_col]

            if l1 not in hierarchy:
                hierarchy[l1] = {}
            if l2 not in hierarchy[l1]:
                hierarchy[l1][l2] = []
            if l3 not in hierarchy[l1][l2]:
                hierarchy[l1][l2].append(l3)

        self.taxonomy_hierarchy = hierarchy

        print(f"✓ Taxonomy loaded:")
        print(f"  L1 categories: {len(hierarchy)}")
        print(f"  L2 categories: {sum(len(l2_dict) for l2_dict in hierarchy.values())}")
        print(f"  L3 categories: {sum(len(l3_list) for l2_dict in hierarchy.values() for l3_list in l2_dict.values())}")

    def add_few_shot_examples(self, examples: List[Dict]):
        """
        Add few-shot examples to improve classification.

        Parameters:
        -----------
        examples : List[Dict]
            List of example classifications, each with:
            {'description': str, 'l1': str, 'l2': str, 'l3': str}
        """
        self.few_shot_examples = examples
        print(f"✓ Added {len(examples)} few-shot examples")

    def _format_taxonomy_for_prompt(self) -> str:
        """Format taxonomy hierarchy as readable string for prompt."""
        lines = []
        for l1, l2_dict in self.taxonomy_hierarchy.items():
            lines.append(f"\n{l1}:")
            for l2, l3_list in l2_dict.items():
                lines.append(f"  └─ {l2}:")
                for l3 in l3_list:
                    lines.append(f"      └─ {l3}")
        return '\n'.join(lines)

    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for prompt."""
        if not self.few_shot_examples:
            return ""

        examples_text = "\n\nHere are some examples:\n"
        for i, ex in enumerate(self.few_shot_examples[:5], 1):  # Limit to 5 examples
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Description: \"{ex['description']}\"\n"
            examples_text += f"Classification: L1={ex['l1']}, L2={ex['l2']}, L3={ex['l3']}\n"

        return examples_text

    def predict(self, description: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Classify a procurement description using LLM.

        Parameters:
        -----------
        description : str
            Procurement description text
        confidence_threshold : float
            Threshold for auto-accept

        Returns:
        --------
        dict : Prediction with L1, L2, L3, confidence, and reasoning
        """
        if self.taxonomy_hierarchy is None:
            raise ValueError("Taxonomy not loaded. Call load_taxonomy() first.")

        # Build prompt
        prompt = f"""You are an expert procurement taxonomy classifier for Greek government contracts.

Your task is to classify the following procurement description into the official 3-level taxonomy (L1 → L2 → L3).

OFFICIAL TAXONOMY:
{self._format_taxonomy_for_prompt()}
{self._format_few_shot_examples()}

CLASSIFICATION RULES:
1. You MUST choose categories that exist in the official taxonomy above
2. L2 must be a child of your chosen L1
3. L3 must be a child of your chosen L2
4. Provide a confidence score (0.0-1.0) based on how certain you are
5. Provide brief reasoning for your classification

PROCUREMENT DESCRIPTION:
"{description}"

Respond with valid JSON only (no other text):
{{
    "l1": "exact L1 category name from taxonomy",
    "l2": "exact L2 category name from taxonomy",
    "l3": "exact L3 category name from taxonomy",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation in Greek or English"
}}"""

        try:
            start_time = time.time()

            # Call LLM
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format='json',
                options={
                    'temperature': self.temperature,
                    'num_predict': 500  # Limit response length
                }
            )

            elapsed = time.time() - start_time

            # Parse response
            result = json.loads(response['response'])

            # Validate predictions exist in taxonomy
            l1 = result.get('l1', '')
            l2 = result.get('l2', '')
            l3 = result.get('l3', '')
            confidence = float(result.get('confidence', 0.0))
            reasoning = result.get('reasoning', 'No reasoning provided')

            # Check validity
            if l1 not in self.taxonomy_hierarchy:
                return {
                    'l1_pred': None,
                    'l2_pred': None,
                    'l3_pred': None,
                    'l1_conf': 0.0,
                    'l2_conf': 0.0,
                    'l3_conf': 0.0,
                    'combined_conf': 0.0,
                    'accept': False,
                    'reason': f'Invalid L1: {l1}',
                    'llm_reasoning': reasoning,
                    'inference_time': elapsed
                }

            if l2 not in self.taxonomy_hierarchy[l1]:
                return {
                    'l1_pred': l1,
                    'l2_pred': None,
                    'l3_pred': None,
                    'l1_conf': confidence,
                    'l2_conf': 0.0,
                    'l3_conf': 0.0,
                    'combined_conf': 0.0,
                    'accept': False,
                    'reason': f'Invalid L2: {l2}',
                    'llm_reasoning': reasoning,
                    'inference_time': elapsed
                }

            if l3 not in self.taxonomy_hierarchy[l1][l2]:
                return {
                    'l1_pred': l1,
                    'l2_pred': l2,
                    'l3_pred': None,
                    'l1_conf': confidence,
                    'l2_conf': confidence,
                    'l3_conf': 0.0,
                    'combined_conf': 0.0,
                    'accept': False,
                    'reason': f'Invalid L3: {l3}',
                    'llm_reasoning': reasoning,
                    'inference_time': elapsed
                }

            # Success
            return {
                'l1_pred': l1,
                'l2_pred': l2,
                'l3_pred': l3,
                'l1_conf': confidence,
                'l2_conf': confidence,
                'l3_conf': confidence,
                'combined_conf': confidence,
                'accept': confidence >= confidence_threshold,
                'reason': 'High confidence' if confidence >= confidence_threshold else 'Low confidence - review needed',
                'llm_reasoning': reasoning,
                'inference_time': elapsed
            }

        except json.JSONDecodeError as e:
            return {
                'l1_pred': None,
                'l2_pred': None,
                'l3_pred': None,
                'l1_conf': 0.0,
                'l2_conf': 0.0,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': f'JSON parse error: {str(e)}',
                'llm_reasoning': 'Invalid response format',
                'inference_time': 0.0
            }
        except Exception as e:
            return {
                'l1_pred': None,
                'l2_pred': None,
                'l3_pred': None,
                'l1_conf': 0.0,
                'l2_conf': 0.0,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': f'Error: {str(e)}',
                'llm_reasoning': 'Prediction failed',
                'inference_time': 0.0
            }


# Example usage
if __name__ == '__main__':
    # Initialize classifier
    classifier = LLMTaxonomyClassifier(model_name='llama3:8b')

    # Load taxonomy
    classifier.load_taxonomy('proc_plan_overall.xlsx', sheet_name='L1-L2-L3')

    # Optional: Add few-shot examples
    examples = [
        {
            'description': 'Προμήθεια ηλεκτρονικών υπολογιστών',
            'l1': 'Τεχνολογία_πληροφορικής_και_τηλεπικοινωνιών',
            'l2': 'Υλικό_υπολογιστών',
            'l3': 'Υπολογιστές'
        }
    ]
    classifier.add_few_shot_examples(examples)

    # Test prediction
    test_desc = "Προμήθεια φορητών υπολογιστών για το γραφείο"
    result = classifier.predict(test_desc, confidence_threshold=0.5)

    print("\nTest Prediction:")
    print(f"L1: {result['l1_pred']}")
    print(f"L2: {result['l2_pred']}")
    print(f"L3: {result['l3_pred']}")
    print(f"Confidence: {result['combined_conf']:.3f}")
    print(f"Reasoning: {result['llm_reasoning']}")
    print(f"Time: {result['inference_time']:.2f}s")
