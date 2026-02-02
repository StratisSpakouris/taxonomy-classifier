# Hierarchical Ensemble Classifier - Complete Methodology

## Overview

This system implements a **sophisticated multi-level ensemble approach** for hierarchical taxonomy classification. Each level (L1, L2, L3) uses a different ensemble strategy optimized for its specific characteristics.

## Architecture

```
                    Procurement Description
                             |
                             v
                    [Text Preprocessing]
                             |
        ┌────────────────────┼────────────────────┐
        |                    |                    |
        v                    v                    v
   [L1 Voting          [L2 Stacking         [L3 XGBoost]
    Ensemble]           Ensemble]
    13 categories       78 categories        210 categories
        |                    |                    |
        └────────────────────┼────────────────────┘
                             v
                 Hierarchical Filtering
                 (L1 → L2 → L3)
                             v
                    Combined Prediction
```

## Level 1: Voting Ensemble

### Strategy
**Soft Voting** with probability averaging across 3 diverse classifiers

### Models
1. **MultinomialNB** (α=0.05)
   - Optimized for text classification
   - Assumes feature independence
   - Fast and efficient

2. **ComplementNB** (α=0.05)
   - Better for imbalanced datasets
   - Estimates parameters from complement of each class
   - Reduces bias for skewed distributions

3. **LogisticRegression** (max_iter=1000)
   - Linear discriminant model
   - Provides calibrated probabilities
   - Regularization prevents overfitting

### Voting Mechanism
```python
P(class) = (P_nb(class) + P_cnb(class) + P_lr(class)) / 3
prediction = argmax(P(class))
```

### Hyperparameter Tuning (Optional)
- **Method**: GridSearchCV with StratifiedKFold (3 folds)
- **Parameters**:
  - `nb__alpha`: [0.01, 0.05, 0.1]
  - `cnb__alpha`: [0.01, 0.05, 0.1]
- **Scoring**: F1-weighted
- **Estimated time**: 1-2 minutes

### Rationale
- **Problem**: 13 categories, relatively easy classification
- **Solution**: Simple voting ensemble provides fast, robust predictions
- **Advantage**: Combines strengths of probabilistic and linear models

## Level 2: Stacking Ensemble

### Strategy
**Stacking** with 4 base models + meta-learner

### Base Models (Layer 1)
1. **MultinomialNB** (α=0.1)
2. **ComplementNB** (α=0.1)
3. **LogisticRegression** (lbfgs solver, L2 penalty)
   - Efficient for moderate-sized problems
   - L2 regularization for smooth decision boundaries
4. **LogisticRegression** (saga solver, L1 penalty)
   - L1 regularization for feature selection
   - Sparse solutions

### Meta-Learner (Layer 2)
**LogisticRegression** learns optimal combination of base model predictions
- Input: Base model probabilities (4 × num_classes features)
- Output: Final prediction
- Training: 3-fold cross-validation to avoid overfitting

### Stacking Process
```
Step 1: Train base models on training data
Step 2: Generate predictions on held-out CV folds
Step 3: Use CV predictions to train meta-learner
Step 4: Final prediction = meta-learner(base_predictions)
```

### Hyperparameter Tuning (Optional)
- **Method**: GridSearchCV (nested CV!)
- **Parameters**:
  - `nb__alpha`: [0.05, 0.1, 0.15]
  - `cnb__alpha`: [0.05, 0.1, 0.15]
- **WARNING**: Very slow (1-3 hours due to nested CV)
- **Default**: Disabled (use fixed hyperparameters)

### Rationale
- **Problem**: 78 categories, moderate difficulty
- **Solution**: Stacking combines diverse models
- **Advantage**: Meta-learner learns which base model to trust for which patterns
- **Trade-off**: More complex but significantly more accurate than voting

## Level 3: XGBoost Classifier

### Strategy
**Gradient Boosting** with decision trees

### Model Configuration
```python
XGBClassifier(
    n_estimators=100,        # Number of trees
    learning_rate=0.1,       # Step size shrinkage
    max_depth=4,             # Tree depth (prevents overfitting)
    subsample=0.8,           # Sample 80% of data per tree
    colsample_bytree=0.8,    # Sample 80% of features per tree
    objective='multi:softprob',  # Multi-class with probabilities
    tree_method='hist',      # CPU-optimized histogram method
    n_jobs=-1                # Use all CPU cores
)
```

### Gradient Boosting Process
```
1. Initialize: F₀(x) = argmax(class probabilities)
2. For m = 1 to 100:
   a. Compute pseudo-residuals
   b. Fit tree to residuals
   c. Update: Fₘ(x) = Fₘ₋₁(x) + learning_rate × tree_m(x)
3. Final prediction = softmax(F₁₀₀(x))
```

### CPU Optimizations
- **Histogram method**: Bins continuous features → faster splits
- **Parallel processing**: Tree building across all cores
- **Memory efficient**: Approximate split finding

### Hyperparameter Tuning (Optional)
- **Method**: GridSearchCV with StratifiedKFold (3 folds)
- **Parameters**:
  - `learning_rate`: [0.05, 0.1, 0.2]
  - `max_depth`: [3, 4, 5]
- **Estimated time**: 3-10 minutes

### Rationale
- **Problem**: 210 categories, most challenging classification
- **Solution**: Powerful gradient boosting
- **Advantage**: Captures complex non-linear relationships
- **Trade-off**: More training time but best accuracy

## Hierarchical Filtering

### Mechanism
```python
# Step 1: Predict L1
l1_pred = L1_ensemble.predict(text)

# Step 2: Filter L2 to valid children of l1_pred
valid_l2 = hierarchy[l1_pred]
l2_probs_filtered = {cls: prob for cls, prob in l2_probs.items() if cls in valid_l2}

# Step 3: Renormalize and predict L2
l2_probs_filtered = normalize(l2_probs_filtered)
l2_pred = argmax(l2_probs_filtered)

# Step 4: Filter L3 to valid children of l2_pred
valid_l3 = hierarchy[l2_pred]
l3_probs_filtered = {cls: prob for cls, prob in l3_probs.items() if cls in valid_l3}

# Step 5: Renormalize and predict L3
l3_probs_filtered = normalize(l3_probs_filtered)
l3_pred = argmax(l3_probs_filtered)
```

### Benefits
1. **Consistency**: Predictions always follow official taxonomy
2. **Accuracy**: Reduces L2/L3 search space dramatically
3. **Interpretability**: Clear parent-child relationships

### Combined Confidence
```python
combined_confidence = P(L1) × P(L2) × P(L3)
auto_accept = combined_confidence >= threshold
```

## Feature Engineering

### TF-IDF Vectorization
```python
TfidfVectorizer(
    max_features=10000,      # Top 10K features (doubled from baseline)
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Ignore rare terms
    lowercase=True,          # Case normalization
    strip_accents=None       # Preserve Greek accents
)
```

### Why TF-IDF?
- **Term Frequency**: Captures importance within document
- **Inverse Document Frequency**: Downweights common terms
- **Bigrams**: Captures phrase-level patterns ("οικοδομικά υλικά")

### Separate Vectorizers
Each level has its own TF-IDF vectorizer:
- **L1**: Learns features discriminative for top-level categories
- **L2**: Learns features specific to mid-level distinctions
- **L3**: Learns fine-grained features for detailed classification

## Performance Characteristics

### Training Time Estimates
| Level | Without GridSearch | With GridSearch |
|-------|-------------------|-----------------|
| L1    | 30 seconds        | 1-2 minutes     |
| L2    | 3-5 minutes       | 1-3 HOURS       |
| L3    | 3-5 minutes       | 3-10 minutes    |
| **Total** | **~10 minutes** | **~1-3.5 hours** |

### Memory Usage
- **TF-IDF matrices**: Sparse representation (~500MB for 15K samples)
- **Models**: ~50-100MB total
- **Peak usage**: ~2-3GB (well within 16GB limit)

### Accuracy Expectations
Based on test set evaluation:
- **L1 Accuracy**: 75-80% (improvement over 72% baseline)
- **L2 Accuracy**: 50-55% (improvement over 45% baseline)
- **L3 Accuracy**: 45-50% (improvement over 40% baseline)
- **Hierarchical L1+L2+L3**: 35-40%

## Configuration Options

### EnsembleConfig Parameters

```python
config = EnsembleConfig()

# TF-IDF
config.max_features = 10000
config.ngram_range = (1, 2)
config.min_df = 2

# L1 (Voting)
config.l1_alpha_nb = 0.05
config.l1_alpha_cnb = 0.05

# L2 (Stacking)
config.l2_alpha_nb = 0.1
config.l2_alpha_cnb = 0.1
config.l2_cv_folds = 3
config.l2_use_calibrated_svc = True  # Slower but more accurate

# L3 (XGBoost)
config.l3_n_estimators = 100
config.l3_learning_rate = 0.1
config.l3_max_depth = 4
config.l3_subsample = 0.8
config.l3_colsample_bytree = 0.8

# GridSearch
config.use_gridsearch = True
config.use_gridsearch_l1 = True
config.use_gridsearch_l2 = False  # SLOW!
config.use_gridsearch_l3 = True
config.gridsearch_cv = 3
config.gridsearch_n_jobs = -1
```

## Usage

### Training
```bash
# Train with ensemble
python main_ensemble.py --input data.xlsx --mode train --use-ensemble

# Train with Naive Bayes (baseline)
python main_ensemble.py --input data.xlsx --mode train
```

### Prediction
```bash
# Auto-detects model type
python main_ensemble.py --input data.xlsx --mode predict --model model.joblib
```

### Python API
```python
from hierarchical_ensemble_classifier import HierarchicalEnsembleClassifier, EnsembleConfig

# Initialize
config = EnsembleConfig()
config.use_gridsearch_l2 = False  # Disable slow L2 GridSearch
classifier = HierarchicalEnsembleClassifier(config=config)

# Train
(train_l1, test_l1), (train_l2, test_l2), (train_l3, test_l3) = classifier.prepare_data(df)
classifier.train(train_l1, train_l2, train_l3)
classifier.load_official_taxonomy('taxonomy.xlsx')

# Predict
result = classifier.predict("Προμήθεια ηλεκτρονικών υπολογιστών", confidence_threshold=0.5)
# Returns: {'l1_pred': '...', 'l2_pred': '...', 'l3_pred': '...',
#           'l1_conf': 0.89, 'l2_conf': 0.67, 'l3_conf': 0.54,
#           'combined_conf': 0.32, 'accept': False, 'reason': '...'}

# Save/Load
classifier.save('ensemble_model.joblib')
classifier = HierarchicalEnsembleClassifier.load('ensemble_model.joblib')
```

## Comparison: Ensemble vs Naive Bayes

| Aspect | Naive Bayes (Baseline) | Ensemble (New) |
|--------|------------------------|----------------|
| **L1 Model** | Single MultinomialNB | Voting (3 models) |
| **L2 Model** | Single MultinomialNB | Stacking (4+1 models) |
| **L3 Model** | Single MultinomialNB | XGBoost (100 trees) |
| **Training Time** | ~5 minutes | ~10 minutes (no GridSearch) |
| **Prediction Speed** | Fast (~100ms/sample) | Moderate (~200ms/sample) |
| **Accuracy** | Baseline | +5-10% improvement |
| **Model Size** | ~20MB | ~80MB |
| **Complexity** | Low | High |
| **Robustness** | Moderate | High |
| **Hyperparameters** | Simple (3 alphas) | Complex (many params) |

## Bugs Fixed

1. ✅ **CalibratedClassifierCV**: Fixed `cv='prefit'` → `cv=2`
2. ✅ **L1 penalty solver**: Fixed `solver='lbfgs'` → `solver='saga'`
3. ✅ **Print statements**: Fixed level labels (L1/L2/L3) and added time units

## Future Enhancements

### Potential Improvements
1. **Feature Engineering**:
   - Add domain-specific features (keywords, patterns)
   - Use pre-trained embeddings (BERT, etc.)

2. **Model Variants**:
   - Try Random Forest for L2
   - Experiment with Neural Networks
   - Add CatBoost as alternative to XGBoost

3. **Ensemble Strategies**:
   - Weighted voting (learn weights)
   - Blending instead of stacking
   - Model selection per category

4. **Optimization**:
   - Parallel training of levels
   - Early stopping for XGBoost
   - Incremental learning for updates

5. **Interpretability**:
   - SHAP values for XGBoost
   - Feature importance analysis
   - Confusion matrix per level

## References

- **Voting**: Kuncheva, L. I. (2004). Combining Pattern Classifiers
- **Stacking**: Wolpert, D. H. (1992). Stacked Generalization
- **XGBoost**: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
- **Hierarchical Classification**: Silla & Freitas (2011). A Survey of Hierarchical Classification

## License

Internal use only.
