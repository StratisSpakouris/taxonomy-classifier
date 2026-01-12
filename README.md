# Procurement Taxonomy Classifier

A hierarchical machine learning system for automatically classifying procurement contracts into a 3-level taxonomy (L1 → L2 → L3).

## Features

- **Hierarchical Classification**: Three-level taxonomy with automatic filtering
- **Confidence-Based Predictions**: Automatic acceptance or flagging for manual review
- **Granular Highlighting**: Excel highlighting at individual taxonomy level
- **Official Taxonomy Integration**: Enforces official hierarchy structure
- **Dual Interface**: Both Web UI and Command-line interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web UI (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Or use the launch script:
```bash
./run_app.sh
```

The UI provides two main functions:

#### Train Model Tab
1. Upload Excel file with labeled training data
2. Configure parameters (test split, confidence threshold, advanced options)
3. Click "Train Model"
4. Download trained model (.joblib file)

#### Make Predictions Tab
1. Upload Excel file with data to classify
2. Upload trained model or use model from training session
3. Configure prediction parameters
4. Click "Generate Predictions"
5. Download results Excel file with predictions and highlighting

### Option 2: Command Line

#### Train a new model:
```bash
python main.py --input proc_plan_overall.xlsx --mode train --threshold 0.5
```

#### Update existing model with new data:
```bash
python main.py --input new_data.xlsx --mode update --threshold 0.5
```

#### Make predictions with existing model:
```bash
python main.py --input data.xlsx --mode predict --model taxonomy_classifier_YYYYMMDD_HHMMSS.joblib --threshold 0.5
```

## Input File Format

Your Excel file must contain a sheet named `Overall` with the following columns:

- `Συνοπτική Περιγραφή Αντικειμένου Σύμβασης`: Text description (required)
- `Επίπεδο Κατηγοριοποίησης L.1`: L1 taxonomy label (for training)
- `Επίπεδο Κατηγοριοποίησης L.2`: L2 taxonomy label (for training)
- `Επίπεδο Κατηγοριοποίησης L.3`: L3 taxonomy label (for training)

For the official taxonomy, include a sheet named `L1-L2-L3` with columns:
- `L1`: Level 1 categories
- `L2`: Level 2 categories
- `L3`: Level 3 categories

## Output

The classifier generates an Excel file with predictions:

- All original columns
- `L1_Prediction`, `L2_Prediction`, `L3_Prediction`: Predicted categories
- `L1_Confidence`, `L2_Confidence`, `L3_Confidence`: Individual confidence scores
- `Combined_Confidence`: Overall prediction confidence
- `Auto_Accept`: Boolean flag for automatic acceptance
- `Review_Reason`: Explanation for manual review requirement

**Granular Highlighting**: Only taxonomy levels with confidence below the threshold are highlighted in yellow, allowing you to see exactly which level needs review.

## Model Performance

The model provides accuracy metrics for each taxonomy level:
- **L1 Accuracy**: ~72% (top-level categories)
- **L2 Accuracy**: ~45% (mid-level categories)
- **L3 Accuracy**: ~40% (fine-grained categories)

Performance varies by category and data quality.

## Project Structure

```
.
├── app.py                      # Streamlit web UI
├── main.py                     # CLI and pipeline orchestration
├── hierarchical_classifier.py  # ML model implementation
├── preprocessing.py            # Data preprocessing and normalization
├── requirements.txt            # Python dependencies
├── run_app.sh                  # Quick launch script
└── README.md                   # This file
```

## How It Works

1. **Data Preprocessing**: Normalizes category names and text descriptions
2. **Hierarchical Training**: Trains separate classifiers for L1, L2, and L3
3. **Official Taxonomy Loading**: Enforces valid parent-child relationships
4. **Prediction**: Predicts L1 first, then filters L2 options, then filters L3 options
5. **Confidence Scoring**: Combines probabilities across all levels
6. **Granular Output**: Highlights only uncertain predictions at specific levels

## Configuration

### Training Parameters

- `--threshold`: Confidence threshold for auto-accept (default: 0.5)
- `alpha_l1/l2/l3`: Smoothing parameters for Naive Bayes (default: 0.1)
- `max_features`: Maximum TF-IDF features (default: 5000)
- `test_size`: Test set proportion (default: 0.2)

### Prediction Parameters

- Confidence threshold (adjustable per run)
- Enable/disable granular highlighting
- Custom output filename

## Troubleshooting

**Issue**: "No L2 children for predicted L1"
- **Cause**: Category name mismatch between training data and official taxonomy
- **Solution**: The normalization function handles this automatically

**Issue**: Low prediction accuracy
- **Cause**: Insufficient or imbalanced training data
- **Solution**: Provide more labeled examples, especially for rare categories

**Issue**: High manual review percentage
- **Cause**: Low confidence threshold or challenging data
- **Solution**: Lower the threshold or improve training data quality

## License

Internal use only.
