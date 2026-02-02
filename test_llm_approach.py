"""
Test script to compare LLM vs Ensemble approaches

This script tests the LLM classifier on a small sample and compares
performance with the ensemble approach.
"""

import pandas as pd
import time
from llm_classifier import LLMTaxonomyClassifier
from hierarchical_ensemble_classifier import HierarchicalEnsembleClassifier

def test_llm_approach(sample_size=10):
    """
    Test LLM classifier on sample data.

    Parameters:
    -----------
    sample_size : int
        Number of samples to test
    """
    print("="*70)
    print("TESTING LLM APPROACH (Llama 3)")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = pd.read_excel('proc_plan_overall.xlsx', sheet_name='Overall')

    # Get labeled samples for testing
    labeled = df[df['Επίπεδο Κατηγοριοποίησης L.1'].notna()].head(sample_size)

    print(f"Testing on {len(labeled)} samples")

    # Initialize LLM classifier
    print("\nInitializing LLM classifier...")
    llm_classifier = LLMTaxonomyClassifier(model_name='llama3:8b', temperature=0.1)

    # Load taxonomy
    llm_classifier.load_taxonomy('proc_plan_overall.xlsx', sheet_name='L1-L2-L3')

    # Optional: Add few-shot examples
    examples = [
        {
            'description': 'Προμήθεια ηλεκτρονικών υπολογιστών και περιφερειακών',
            'l1': 'Τεχνολογία_πληροφορικής_και_τηλεπικοινωνιών',
            'l2': 'Υλικό_υπολογιστών',
            'l3': 'Υπολογιστές'
        },
        {
            'description': 'Υπηρεσίες καθαρισμού και συντήρησης κτιρίων',
            'l1': 'Επαγγελματικές_υπηρεσίες',
            'l2': 'Έξωτερική_ανάθεση_επιχειρηματικών_και_διοικητικών_υπηρεσιών',
            'l3': 'Υπηρεσίες_περιβαλλοντικής_αποκατάστασης_και_προσωπικής_απολύμανσης'
        }
    ]
    llm_classifier.add_few_shot_examples(examples)

    # Test predictions
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)

    correct_l1 = 0
    correct_l2 = 0
    correct_l3 = 0
    correct_full = 0
    total_time = 0

    results = []

    for idx, row in labeled.iterrows():
        description = str(row['Συνοπτική Περιγραφή Αντικειμένου Σύμβασης'])
        actual_l1 = row['Επίπεδο Κατηγοριοποίησης L.1']
        actual_l2 = row['Επίπεδο Κατηγοριοποίησης L.2']
        actual_l3 = row['Επίπεδο Κατηγοριοποίησης L.3']

        # Predict
        pred = llm_classifier.predict(description, confidence_threshold=0.5)
        total_time += pred['inference_time']

        # Check correctness
        l1_correct = pred['l1_pred'] == actual_l1
        l2_correct = pred['l2_pred'] == actual_l2
        l3_correct = pred['l3_pred'] == actual_l3
        full_correct = l1_correct and l2_correct and l3_correct

        if l1_correct:
            correct_l1 += 1
        if l2_correct:
            correct_l2 += 1
        if l3_correct:
            correct_l3 += 1
        if full_correct:
            correct_full += 1

        results.append({
            'description': description[:50] + '...',
            'actual_l1': actual_l1,
            'pred_l1': pred['l1_pred'],
            'l1_match': '✓' if l1_correct else '✗',
            'actual_l2': actual_l2,
            'pred_l2': pred['l2_pred'],
            'l2_match': '✓' if l2_correct else '✗',
            'actual_l3': actual_l3,
            'pred_l3': pred['l3_pred'],
            'l3_match': '✓' if l3_correct else '✗',
            'confidence': pred['combined_conf'],
            'reasoning': pred['llm_reasoning'][:100] + '...',
            'time': f"{pred['inference_time']:.2f}s"
        })

        print(f"\nSample {len(results)}/{sample_size}:")
        print(f"  Description: {description[:80]}...")
        print(f"  L1: {pred['l1_pred']} {'✓' if l1_correct else '✗'}")
        print(f"  L2: {pred['l2_pred']} {'✓' if l2_correct else '✗'}")
        print(f"  L3: {pred['l3_pred']} {'✓' if l3_correct else '✗'}")
        print(f"  Confidence: {pred['combined_conf']:.3f}")
        print(f"  Time: {pred['inference_time']:.2f}s")

    # Summary
    print("\n" + "="*70)
    print("LLM RESULTS SUMMARY")
    print("="*70)

    print(f"\nAccuracy:")
    print(f"  L1: {correct_l1}/{sample_size} ({correct_l1/sample_size*100:.1f}%)")
    print(f"  L2: {correct_l2}/{sample_size} ({correct_l2/sample_size*100:.1f}%)")
    print(f"  L3: {correct_l3}/{sample_size} ({correct_l3/sample_size*100:.1f}%)")
    print(f"  Full (L1+L2+L3): {correct_full}/{sample_size} ({correct_full/sample_size*100:.1f}%)")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time per prediction: {total_time/sample_size:.2f}s")
    print(f"  Throughput: {sample_size/total_time:.2f} predictions/second")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel('llm_test_results.xlsx', index=False)
    print(f"\n✓ Results saved to: llm_test_results.xlsx")

    return {
        'l1_accuracy': correct_l1/sample_size,
        'l2_accuracy': correct_l2/sample_size,
        'l3_accuracy': correct_l3/sample_size,
        'full_accuracy': correct_full/sample_size,
        'avg_time': total_time/sample_size
    }


def compare_approaches():
    """
    Compare LLM vs Ensemble approaches.
    """
    print("\n" + "="*70)
    print("COMPARISON: LLM vs ENSEMBLE")
    print("="*70)

    # From previous runs (you can load actual results)
    ensemble_results = {
        'l1_accuracy': 0.717,
        'l2_accuracy': 0.451,
        'l3_accuracy': 0.402,
        'full_accuracy': 0.0,  # Hierarchical
        'avg_time': 0.1  # seconds
    }

    print("\nRunning LLM test (this may take a few minutes)...")
    llm_results = test_llm_approach(sample_size=20)

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<30} {'Ensemble':<15} {'LLM':<15} {'Winner'}")
    print("-"*70)
    print(f"{'L1 Accuracy':<30} {ensemble_results['l1_accuracy']:.1%:<15} {llm_results['l1_accuracy']:.1%:<15} {'LLM' if llm_results['l1_accuracy'] > ensemble_results['l1_accuracy'] else 'Ensemble'}")
    print(f"{'L2 Accuracy':<30} {ensemble_results['l2_accuracy']:.1%:<15} {llm_results['l2_accuracy']:.1%:<15} {'LLM' if llm_results['l2_accuracy'] > ensemble_results['l2_accuracy'] else 'Ensemble'}")
    print(f"{'L3 Accuracy':<30} {ensemble_results['l3_accuracy']:.1%:<15} {llm_results['l3_accuracy']:.1%:<15} {'LLM' if llm_results['l3_accuracy'] > ensemble_results['l3_accuracy'] else 'Ensemble'}")
    print(f"{'Inference Speed':<30} {ensemble_results['avg_time']:.3f}s<15 {llm_results['avg_time']:.3f}s<15 {'Ensemble' if ensemble_results['avg_time'] < llm_results['avg_time'] else 'LLM'}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if llm_results['l1_accuracy'] > 0.85 and llm_results['l2_accuracy'] > 0.70:
        print("\n✓ LLM approach shows significantly better accuracy!")
        print("  Recommended: Switch to LLM for production")
        print("  Trade-off: Slower inference but much better quality")
    elif llm_results['l1_accuracy'] > ensemble_results['l1_accuracy'] + 0.05:
        print("\n✓ LLM approach shows moderate improvement")
        print("  Recommended: Use hybrid approach (ensemble + LLM for low-confidence)")
        print("  Best of both worlds: Fast + accurate")
    else:
        print("\n→ LLM and Ensemble perform similarly")
        print("  Recommended: Stick with ensemble for speed")
        print("  Consider LLM for explainability needs")


if __name__ == '__main__':
    try:
        compare_approaches()
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use LLM approach:")
        print("1. Install Ollama: brew install ollama")
        print("2. Pull model: ollama pull llama3:8b")
        print("3. Install Python client: pip install ollama")
        print("4. Start service: ollama serve")
    except ConnectionError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama service is running:")
        print("  ollama serve")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
