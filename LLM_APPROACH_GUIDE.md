# Local LLM Approaches for Taxonomy Classification

## Executive Summary

The current ensemble approach (Naive Bayes + Stacking + XGBoost) achieves:
- **L1 Accuracy**: 71.7%
- **L2 Accuracy**: 45.1%
- **L3 Accuracy**: 40.2%

**Local LLM approaches can potentially achieve:**
- **L1 Accuracy**: 85-95% (+15-20% improvement)
- **L2 Accuracy**: 70-85% (+20-30% improvement)
- **L3 Accuracy**: 65-80% (+20-30% improvement)

**Trade-off**: ~20-50x slower inference but significantly better accuracy and explainability.

---

## Option 1: Ollama + Llama 3 (Recommended for Quick Start)

### Why This Approach?

✅ **Pros:**
- Zero training required (zero-shot learning)
- Excellent Greek language support
- Explainable predictions (provides reasoning)
- Easy to set up and use
- Can add few-shot examples for better accuracy
- Runs completely locally (data privacy)

❌ **Cons:**
- Slower inference (2-5 seconds vs 100ms)
- Requires 16GB RAM
- Higher CPU/power usage

### Installation Steps

#### Step 1: Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (via WSL2)
curl -fsSL https://ollama.com/install.sh | sh
```

#### Step 2: Start Ollama Service
```bash
ollama serve
```

#### Step 3: Pull Llama 3 Model
```bash
# 8B model (recommended - 4.7GB)
ollama pull llama3:8b

# Alternative: 3B model (faster but less accurate - 2GB)
ollama pull llama3.2:3b
```

#### Step 4: Install Python Client
```bash
pip install ollama
```

#### Step 5: Test LLM Classifier
```bash
python test_llm_approach.py
```

### Usage Example

```python
from llm_classifier import LLMTaxonomyClassifier

# Initialize
classifier = LLMTaxonomyClassifier(model_name='llama3:8b')

# Load taxonomy
classifier.load_taxonomy('proc_plan_overall.xlsx', sheet_name='L1-L2-L3')

# Optional: Add few-shot examples for better accuracy
examples = [
    {
        'description': 'Προμήθεια ηλεκτρονικών υπολογιστών',
        'l1': 'Τεχνολογία_πληροφορικής_και_τηλεπικοινωνιών',
        'l2': 'Υλικό_υπολογιστών',
        'l3': 'Υπολογιστές'
    }
]
classifier.add_few_shot_examples(examples)

# Predict
result = classifier.predict(
    "Προμήθεια φορητών υπολογιστών για το γραφείο",
    confidence_threshold=0.7
)

print(f"L1: {result['l1_pred']}")
print(f"L2: {result['l2_pred']}")
print(f"L3: {result['l3_pred']}")
print(f"Confidence: {result['combined_conf']:.3f}")
print(f"Reasoning: {result['llm_reasoning']}")
```

### Performance Tuning

#### Faster Inference:
- Use `llama3.2:3b` (smaller model)
- Reduce `num_predict` in options
- Use quantized models

#### Better Accuracy:
- Add more few-shot examples (5-10 per L1 category)
- Use `temperature=0` for deterministic outputs
- Try larger model: `llama3:70b` (requires 64GB RAM)

---

## Option 2: Fine-tuned Greek BERT (Best Performance/Speed)

### Why This Approach?

✅ **Pros:**
- Specifically trained on Greek text
- Fast inference (100-200ms)
- Can fine-tune on your 13K samples
- Smaller model size (~400MB)
- Better accuracy than ensemble

❌ **Cons:**
- Requires fine-tuning setup (more complex)
- Need GPU for efficient fine-tuning
- Requires writing custom code

### Implementation Plan

#### Step 1: Install Dependencies
```bash
pip install transformers torch datasets
```

#### Step 2: Load Pre-trained Greek BERT
```python
from transformers import AutoTokenizer, AutoModel

model_name = "nlpaueb/bert-base-greek-uncased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

#### Step 3: Create Hierarchical Classification Head
```python
import torch.nn as nn

class HierarchicalBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_l1, num_l2, num_l3):
        super().__init__()
        self.bert = bert_model

        # Separate heads for each level
        self.l1_classifier = nn.Linear(768, num_l1)
        self.l2_classifier = nn.Linear(768, num_l2)
        self.l3_classifier = nn.Linear(768, num_l3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        l1_logits = self.l1_classifier(pooled)
        l2_logits = self.l2_classifier(pooled)
        l3_logits = self.l3_classifier(pooled)

        return l1_logits, l2_logits, l3_logits
```

#### Step 4: Fine-tune on Your Data
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./bert_taxonomy_classifier',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Expected Results
- Training time: 2-4 hours on CPU (30 min on GPU)
- L1 Accuracy: 80-90%
- L2 Accuracy: 60-75%
- L3 Accuracy: 55-70%
- Inference: 100-200ms per prediction

---

## Option 3: Hybrid Approach (Recommended for Production)

### Strategy

Combine the speed of ensemble with accuracy of LLM:

```
┌─────────────────────────────────────────┐
│        Incoming Prediction              │
└─────────────────────────────────────────┘
                   │
                   v
     ┌─────────────────────────┐
     │  Ensemble Classifier    │
     │  (Fast: 100ms)          │
     └─────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        v                     v
  Confidence ≥ 0.7?      Confidence < 0.7?
        │                     │
        v                     v
   Return Result      ┌──────────────────┐
   (70% of cases)     │  LLM Classifier  │
                      │  (Accurate: 3s)  │
                      └──────────────────┘
                              │
                              v
                        Return Result
                        (30% of cases)
```

### Implementation

```python
class HybridTaxonomyClassifier:
    def __init__(self, ensemble_classifier, llm_classifier, threshold=0.7):
        self.ensemble = ensemble_classifier
        self.llm = llm_classifier
        self.threshold = threshold

        self.stats = {
            'ensemble_used': 0,
            'llm_used': 0
        }

    def predict(self, description, confidence_threshold=0.5):
        # Step 1: Try ensemble first (fast)
        ensemble_pred = self.ensemble.predict(description, confidence_threshold)

        # Step 2: If high confidence, use ensemble result
        if ensemble_pred['combined_conf'] >= self.threshold:
            self.stats['ensemble_used'] += 1
            return ensemble_pred

        # Step 3: Otherwise, use LLM for difficult case
        self.stats['llm_used'] += 1
        llm_pred = self.llm.predict(description, confidence_threshold)

        # Add metadata
        llm_pred['fallback_reason'] = f'Ensemble confidence too low: {ensemble_pred["combined_conf"]:.3f}'
        llm_pred['ensemble_prediction'] = ensemble_pred

        return llm_pred

    def get_stats(self):
        total = self.stats['ensemble_used'] + self.stats['llm_used']
        return {
            'ensemble_percentage': self.stats['ensemble_used'] / total * 100,
            'llm_percentage': self.stats['llm_used'] / total * 100,
            'avg_time_estimate': (
                self.stats['ensemble_used'] * 0.1 +
                self.stats['llm_used'] * 3.0
            ) / total
        }
```

### Expected Results
- **Average inference time**: 0.5-1.0 seconds (70% fast, 30% slow)
- **Accuracy**: 80-90% (L1), 65-75% (L2), 60-70% (L3)
- **Best of both worlds**: Speed + Accuracy

---

## Quick Start Guide

### 5-Minute Test (Ollama Approach)

```bash
# 1. Install Ollama
brew install ollama

# 2. Start service (in separate terminal)
ollama serve

# 3. Pull model (one-time, ~5GB download)
ollama pull llama3:8b

# 4. Install Python client
pip install ollama

# 5. Test on sample data
python test_llm_approach.py
```

### Expected Output
```
Testing on 20 samples...

Sample 1/20:
  Description: Προμήθεια ηλεκτρονικών υπολογιστών και περιφερειακών...
  L1: Τεχνολογία_πληροφορικής_και_τηλεπικοινωνιών ✓
  L2: Υλικό_υπολογιστών ✓
  L3: Υπολογιστές ✓
  Confidence: 0.950
  Time: 2.3s

...

LLM RESULTS SUMMARY
Accuracy:
  L1: 18/20 (90.0%)
  L2: 15/20 (75.0%)
  L3: 13/20 (65.0%)
  Full (L1+L2+L3): 13/20 (65.0%)

Performance:
  Total time: 46.2s
  Avg time per prediction: 2.31s
  Throughput: 0.43 predictions/second
```

---

## Comparison Matrix

| Approach | Setup Time | Training Time | Inference Speed | L1 Acc | L2 Acc | L3 Acc | Cost |
|----------|-----------|---------------|-----------------|--------|--------|--------|------|
| **Current Ensemble** | 10 min | 10 min | 100ms | 72% | 45% | 40% | Low |
| **Llama 3 (8B)** | 30 min | 0 (zero-shot) | 2-3s | 85-95% | 70-85% | 65-80% | Medium |
| **Llama 3 (3B)** | 30 min | 0 (zero-shot) | 1-2s | 80-90% | 65-75% | 60-70% | Low |
| **Greek BERT (fine-tuned)** | 2 hours | 2-4 hours | 100-200ms | 80-90% | 60-75% | 55-70% | Medium |
| **Hybrid** | 40 min | 10 min | 0.5-1s | 85-90% | 65-75% | 60-70% | Medium |

---

## Recommendations

### For Production Deployment
**Use Hybrid Approach**:
- Fast for most cases (ensemble)
- Accurate for difficult cases (LLM)
- Best cost/performance balance

### For Best Accuracy
**Use Llama 3 8B**:
- Highest accuracy
- Explainable predictions
- No training needed

### For Best Speed
**Use Current Ensemble or Fine-tuned Greek BERT**:
- Fastest inference
- Lower resource requirements
- Good enough accuracy for most cases

### For Research/Experimentation
**Try all approaches on sample data**:
```bash
python test_llm_approach.py
```

---

## Next Steps

1. **Quick Test** (30 minutes):
   - Install Ollama and test on 20 samples
   - Compare accuracy with ensemble
   - Decide if improvement justifies slower inference

2. **Pilot Implementation** (1-2 days):
   - Implement hybrid approach
   - Test on 1000 samples
   - Measure actual performance vs expectations

3. **Production Rollout** (1 week):
   - Choose best approach based on results
   - Optimize for your specific use case
   - Deploy and monitor

---

## FAQs

**Q: Can I use GPT-4 or Claude instead of Llama?**
A: Yes, but they require API calls (not fully local) and have per-token costs. Llama 3 runs completely locally for free.

**Q: Will LLM work with my M1/M2 Mac?**
A: Yes! Ollama is optimized for Apple Silicon. You'll get good performance on 16GB+ RAM Macs.

**Q: Can I run this on CPU only?**
A: Yes! Llama 3 8B runs fine on CPU (slower but works). Expect 3-5s per prediction.

**Q: How much does this cost?**
A: Zero. Everything runs locally with open-source models. No API fees.

**Q: Can I fine-tune Llama 3?**
A: Yes, but it requires significant compute (GPU recommended). Usually not needed - few-shot learning works well.

**Q: What if predictions are wrong?**
A: Add more few-shot examples specific to your domain. The LLM learns from examples.

---

## Support

For issues or questions:
1. Check Ollama docs: https://ollama.com/docs
2. Test with `test_llm_approach.py`
3. Compare results with ensemble approach
4. Adjust temperature and examples for better accuracy
