import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from preprocessing import TextPreprocessor, DataPreprocessor, normalize_category_name
import joblib
import json
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

class HierachicalTaxonomyClassifier:
    """
    1. First predict L1 (13 options)
    2. Then predict L2 (only 5-10 options per L1)
    3. Finally predict L3 (only 3-8 options per L2)

    Why Multinomial Naive Bayes?
    - Designed for text
    - Handles class imbalance via priors
    - Fast training and prediction
    - Probabilistic outputs (confidence score)
    - Performs well on document classification
    """

    def __init__(self,
                 alpha_l1=0.1,
                 alpha_l2=0.1,
                 alpha_l3=0.1,
                 max_features=5000,
                 ngram_range=(1,2),
                 min_df=2,
                 text_preprocessor=None):
        
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.alpha_l3 = alpha_l3

        # TD-IDF Vectorizers for each level
        self.vectorizer_l1 = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents=None
        )

        self.vectorizer_l2 = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents=None
        )

        self.vectorizer_l3 = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents=None
        )

        # Models for each level
        self.model_l1 = None
        self.model_l2 = None
        self.model_l3 = None

        # Hierarchy
        self.hierarchy_l1_to_l2 = defaultdict(set)
        self.hierarchy_l2_to_l3 = defaultdict(set)

        # Stats
        self.training_stats = {}

    def load_official_taxonomy(self, excel_path, sheet_name='L1-L2-L3'):
        """
        Load official taxonomy from Excel file and normalize category names
        """

        print(f"\nLoading official taxonomy from '{sheet_name}' sheet...")

        # Read taxonomy sheet
        df_tax = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Detect column names
        l1_col = 'L1' if 'L1' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.1'
        l2_col = 'L2' if 'L2' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.2'
        l3_col = 'L3' if 'L3' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.3'

        df_tax[l1_col] = df_tax[l1_col].ffill()
        df_tax[l2_col] = df_tax[l2_col].ffill()

        # Clean the data
        df_tax = df_tax.dropna(subset=[l3_col])

        # IMPORTANT: Normalize all category names to match training data format
        df_tax[l1_col] = df_tax[l1_col].apply(normalize_category_name)
        df_tax[l2_col] = df_tax[l2_col].apply(normalize_category_name)
        df_tax[l3_col] = df_tax[l3_col].apply(normalize_category_name)

        # Build official hierarchy
        self.official_hierarchy_l1_to_l2 = {}
        self.official_hierarchy_l2_to_l3 = {}

        for _, row in df_tax.iterrows():
            l1 = row[l1_col]
            l2 = row[l2_col]
            l3 = row[l3_col]

            # L1 -> L2 mapping
            if l1 not in self.official_hierarchy_l1_to_l2:
                self.official_hierarchy_l1_to_l2[l1] = set()
            self.official_hierarchy_l1_to_l2[l1].add(l2)

            # L2 -> L3 mapping
            if l2 not in self.official_hierarchy_l2_to_l3:
                self.official_hierarchy_l2_to_l3[l2] = set()
            self.official_hierarchy_l2_to_l3[l2].add(l3)

        # Get all valid categories
        self.official_l1_categories = set(df_tax[l1_col].unique())
        self.official_l2_categories = set(df_tax[l2_col].unique())
        self.official_l3_categories = set(df_tax[l3_col])

        print(f"    Official taxonomy loaded:")
        print(f"    L1 categories: {len(self.official_l1_categories)}")
        print(f"    L2 categories: {len(self.official_l2_categories)}")
        print(f"    L3 categories: {len(self.official_l3_categories)}")

        # Use official hierarchy for predictions
        self.hierarchy_l1_to_l2 = self.official_hierarchy_l1_to_l2
        self.hierarchy_l2_to_l3 = self.official_hierarchy_l2_to_l3

        print(f"    Hierarchy updated with official taxonomy")
        print(f"    L1->L2 mappings: {len(self.official_hierarchy_l1_to_l2)}")
        print(f"    L2->L3 mappings: {len(self.official_hierarchy_l2_to_l3)}")

        # Show sample of hierarchy
        print(f"\nSample hierarchy:")
        for i, (l1, l2_set) in enumerate(list(self.official_hierarchy_l1_to_l2.items())):
            print(f"    {l1} -> {len(l2_set)} L2 categories")

        for i, (l2, l3_set) in enumerate(list(self.official_hierarchy_l2_to_l3.items())):
            print(f"    {l2} -> {len(l3_set)} L3 categories")

        return df_tax

    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        This method delegates to preprocessing.py
        """

        dataset_prep = DataPreprocessor(text_preprocessor=self.text_preprocessor)

        result = dataset_prep.prepare_dataset(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # Extract prepared datasets
        train_l1 = result['train_l1']
        test_l1 = result['test_l1']
        train_l2 = result['train_l2']  # FIX: Use correct L2 datasets
        test_l2 = result['test_l2']     # FIX: Use correct L2 datasets
        train_l3 = result['train_l3']  # FIX: Use correct L3 datasets
        test_l3 = result['test_l3']     # FIX: Use correct L3 datasets

        # Store hierarhcy mapping
        for l1, l2_set, in result['hierarchy_l1_to_l2'].items():
            self.hierarchy_l1_to_l2[l1].update(l2_set)

        for l2, l3_set, in result['hierarchy_l2_to_l3'].items():
            self.hierarchy_l2_to_l3[l2].update(l3_set)

        # Store preprocessing stats
        self.training_stats['preprocessing'] = result['stats']

        return (train_l1, test_l1), (train_l2, test_l2), (train_l3, test_l3)
    
    def train(self, train_l1, train_l2, train_l3):

        print("="*70)
        print("TRAINING HIERARCHICAL CLASSIFIER")
        print("="*70)

        # Train L1 Model
        print("\n[1/3] Training L1 classifier...")
        X_train_l1 = self.vectorizer_l1.fit_transform(train_l1['description_clean'])
        y_train_l1 = train_l1['Επίπεδο Κατηγοριοποίησης L.1']
        self.model_l1 = MultinomialNB(alpha=self.alpha_l1)
        self.model_l1.fit(X_train_l1, y_train_l1)

        # Train L2 Model
        print("\n[2/3] Training L2 classifier...")
        X_train_l2 = self.vectorizer_l2.fit_transform(train_l2['description_clean'])
        y_train_l2 = train_l2['Επίπεδο Κατηγοριοποίησης L.2']  # FIX: Use actual L2 labels, not boolean

        # Validate: Check for NaN values before training
        nan_count_l2 = y_train_l2.isna().sum()
        if nan_count_l2 > 0:
            raise ValueError(
                f"L2 training data contains {nan_count_l2} NaN values!"
            )

        self.model_l2 = MultinomialNB(alpha=self.alpha_l2)
        self.model_l2.fit(X_train_l2, y_train_l2)

        # Train L3 Model
        print("\n[3/3] Training L3 classifier...")
        X_train_l3 = self.vectorizer_l3.fit_transform(train_l3['description_clean'])
        y_train_l3 = train_l3['Επίπεδο Κατηγοριοποίησης L.3']  # FIX: Use actual L3 labels, not boolean
        self.model_l3 = MultinomialNB(alpha=self.alpha_l3)
        self.model_l3.fit(X_train_l3, y_train_l3)

        # Store training stats
        #self.training_stats.update({...})

        print("\n" + "="*70)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)

    def predict(self, description, confidence_threshold=0.5):
        """
        Predict taxonomy for a new procurement request.

        How it works:
        1. Preprocess text usigng TextPreprocessor
        2. Predict L1 with confidence
        3. Filter L2 options to children of predicted L1
        4. Predict L2 with confidence
        5. Filter L3 options to children of predicted L2
        6. Predict L3 with confidence
        7. Calculate combined confidence = P(L1) x P(L2) x P(L3) 
        """

        text_clean = self.text_preprocessor.preprocess(description)

        # Predict L1
        X_l1 = self.vectorizer_l1.transform([text_clean])
        l1_probs = self.model_l1.predict_proba(X_l1)[0]
        l1_pred_idx = np.argmax(l1_probs)
        l1_pred = self.model_l1.classes_[l1_pred_idx]
        l1_conf = l1_probs[l1_pred_idx]

        # =========== PREDICT L2 (filtered by L1) ===========
        # Get valid L2 categories for this L1
        valid_l2 = self.hierarchy_l1_to_l2.get(l1_pred, set())

        if len(valid_l2) == 0:
            return {
                'l1_pred': l1_pred,
                'l2_pred': None,
                'l3_pred': None,
                'l1_conf': float(l1_conf),
                'l2_conf': 0.0,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': '"No L2 children for predicted L1."'
            }
        
        # Get L2 predictions
        X_l2 = self.vectorizer_l2.transform([text_clean])
        l2_probs_all = self.model_l2.predict_proba(X_l2)[0]

        # Filter to only valid L2 categories
        l2_filtered_probs = []
        l2_filtered_classes = []
        for i, cls in enumerate(self.model_l2.classes_):
            if cls in valid_l2:
                l2_filtered_probs.append(l2_probs_all[i])
                l2_filtered_classes.append(cls)

        if len(l2_filtered_probs) == 0:
            #print("Predicted L1 has no valid L2 children in the training data.")
            l2_pred_idx = np.argmax(l2_probs_all)
            l2_pred = self.model_l2.classes_[l2_pred_idx]
            l2_conf = l2_probs_all[l2_pred_idx]
        else:
            # Renormalize probablities after filtering
            l2_filtered_probs = np.array(l2_filtered_probs)
            l2_filtered_probs = l2_filtered_probs / l2_filtered_probs.sum()

            # Get the best L2 prediction
            l2_pred_idx = np.argmax(l2_filtered_probs)
            l2_pred = l2_filtered_classes[l2_pred_idx]  # FIX: Use class name, not probability
            l2_conf = l2_filtered_probs[l2_pred_idx]

        # =========== PREDICT L3 (filtered by L2) ===========
        # Get valid L2 categories for this L1
        valid_l3 = self.hierarchy_l2_to_l3.get(l2_pred, set())

        if len(valid_l3) == 0:
            return {
                'l1_pred': l1_pred,
                'l2_pred': None,
                'l3_pred': None,
                'l1_conf': float(l1_conf),
                'l2_conf': 0.0,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': '"No L3 children for predicted L2."'
            }
        
        # Get L2 predictions
        X_l3 = self.vectorizer_l3.transform([text_clean])
        l3_probs_all = self.model_l3.predict_proba(X_l3)[0]

        # Filter to only valid L2 categories
        l3_filtered_probs = []
        l3_filtered_classes = []
        for i, cls in enumerate(self.model_l3.classes_):
            if cls in valid_l3:
                l3_filtered_probs.append(l3_probs_all[i])
                l3_filtered_classes.append(cls)

        if len(l3_filtered_probs) == 0:
            #print("Predicted L2 has no known L3 children in the training data.")
            l3_pred_idx = np.argmax(l3_probs_all)
            l3_pred = self.model_l3.classes_[l3_pred_idx]
            l3_conf = l3_probs_all[l3_pred_idx]
        else:
            # Renormalize probablities after filtering
            l3_filtered_probs = np.array(l3_filtered_probs)
            l3_filtered_probs = l3_filtered_probs / l3_filtered_probs.sum()

            # Get the best L3 prediction
            l3_pred_idx = np.argmax(l3_filtered_probs)
            l3_pred = l3_filtered_classes[l3_pred_idx]  # FIX: Use class name, not probability
            l3_conf = l3_filtered_probs[l3_pred_idx]

        # =========== CALCULATE COMBINED CONFIDENCE ===========
        combined_conf = l1_conf * l2_conf * l3_conf

        result = {
            'l1_pred': l1_pred,
            'l2_pred': l2_pred,
            'l3_pred': l3_pred,
            'l1_conf': float(l1_conf),
            'l2_conf': float(l2_conf),
            'l3_conf': float(l3_conf),
            'combined_conf': float(combined_conf),
            'accept': combined_conf >= confidence_threshold,
            'reason': 'High confidence' if combined_conf >= confidence_threshold else 'Low confidence - review needed'
        }

        # Sanity check before returning
        if not isinstance(result, dict):
            raise TypeError(f"predict() returning wrong type: {type(result)}")

        return result
    
    def evaluate(self, test_l1, test_l2, test_l3):
        """
        Evaluate the model on test data
        """

        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

        results = {}

        # ============== Evaluate L1 ==============
        print("\n[1/3] Evaluating L1 classifier...")
        X_test_l1 = self.vectorizer_l1.transform(test_l1['description_clean'])
        y_test_l1 = test_l1['Επίπεδο Κατηγοριοποίησης L.1']  # FIX: Use actual labels
        y_pred_l1 = self.model_l1.predict(X_test_l1)

        print("\nClassification Report")
        print(classification_report(y_test_l1, y_pred_l1, zero_division=0))

        l1_accuracy = (y_pred_l1 == y_test_l1).mean()
        l1_f1_macro = f1_score(y_test_l1, y_pred_l1, average='macro', zero_division=0)
        l1_f1_weighted = f1_score(y_test_l1, y_pred_l1, average='weighted', zero_division=0)

        results['l1'] = {
            'accuracy': l1_accuracy,
            'f1_macro': l1_f1_macro,
            'f1_weighted': l1_f1_weighted
        }

        # ============== Evaluate L2 ==============
        print("\n[2/3] Evaluating L2 classifier...")
        X_test_l2 = self.vectorizer_l2.transform(test_l2['description_clean'])
        y_test_l2 = test_l2['Επίπεδο Κατηγοριοποίησης L.2']  # FIX: Use actual labels
        y_pred_l2 = self.model_l2.predict(X_test_l2)

        print("\nClassification Report")
        print(classification_report(y_test_l2, y_pred_l2, zero_division=0))

        l2_accuracy = (y_pred_l2 == y_test_l2).mean()
        l2_f1_macro = f1_score(y_test_l2, y_pred_l2, average='macro', zero_division=0)
        l2_f1_weighted = f1_score(y_test_l2, y_pred_l2, average='weighted', zero_division=0)

        results['l2'] = {
            'accuracy': l2_accuracy,
            'f1_macro': l2_f1_macro,
            'f1_weighted': l2_f1_weighted
        }

        # ============== Evaluate L3 ==============
        print("\n[3/3] Evaluating L3 classifier...")
        X_test_l3 = self.vectorizer_l3.transform(test_l3['description_clean'])
        y_test_l3 = test_l3['Επίπεδο Κατηγοριοποίησης L.3']  # FIX: Use actual labels
        y_pred_l3 = self.model_l3.predict(X_test_l3)

        print("\nClassification Report")
        print(classification_report(y_test_l3, y_pred_l3, zero_division=0))

        l3_accuracy = (y_pred_l3 == y_test_l3).mean()
        l3_f1_macro = f1_score(y_test_l3, y_pred_l3, average='macro', zero_division=0)
        l3_f1_weighted = f1_score(y_test_l3, y_pred_l3, average='weighted', zero_division=0)

        results['l3'] = {
            'accuracy': l3_accuracy,
            'f1_macro': l3_f1_macro,
            'f1_weighted': l3_f1_weighted
        }

        # ========= HIERARCHICAL EVALUATION =========
        print("\n[HIERARCHICAL] Evaluating full pipeline...")

        # Test hierarchical prediction with filtering
        correct_full = 0
        correct_l1 = 0
        correct_l2 = 0

        for idx, row in test_l3.iterrows():
            description = row['Συνοπτική Περιγραφή Αντικειμένου Σύμβασης']
            if pd.isna(description):
                description = ""
            else:
                description = str(description)

            try:
                pred = self.predict(description)

                if not isinstance(pred, dict):
                    continue

                actual_l1 = str(row['Επίπεδο Κατηγοριοποίσης L.1'])
                actual_l2 = str(row['Επίπεδο Κατηγοριοποίσης L.2'])
                actual_l3 = str(row['Επίπεδο Κατηγοριοποίσης L.3'])

                # Check L1 correctness
                if pred['l1_pred'] == actual_l1:
                    correct_l1 += 1

                    # Check L2 correctness (only if L1 was correct)
                    if pred['l2_pred'] == actual_l2:
                        correct_l2 += 1

                        # Check L3 correctness (only if L1 and L2 were correct)
                        if pred['l3_pred'] == actual_l3:
                            correct_full += 1
            except Exception as e:
                continue

        hierarchical_l1_acc = correct_l1 / len(test_l3)
        hierarchical_l2_acc = correct_l2 / len(test_l3)
        hierarchical_full_acc = correct_full / len(test_l3)

        results['hierarchical'] = {
            'l1_accuracy': hierarchical_l1_acc,
            'l2_accuracy': hierarchical_l2_acc,
            'full_accuracy': hierarchical_full_acc,
        }

        #========= SUMMARY ========#
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nIndividual Level Performance")
        print(f"    L1 Accuracy: {l1_accuracy:.3f} | F1-Macro: {l1_f1_macro:.3f} | F1-Weighted: {l1_f1_weighted:.3f}")
        print(f"    L2 Accuracy: {l2_accuracy:.3f} | F1-Macro: {l2_f1_macro:.3f} | F1-Weighted: {l2_f1_weighted:.3f}")
        print(f"    L3 Accuracy: {l3_accuracy:.3f} | F1-Macro: {l3_f1_macro:.3f} | F1-Weighted: {l3_f1_weighted:.3f}")

        print(f"\nHierarchical Pipeline Performance")
        print(f"    L1 Correct: {hierarchical_l1_acc:.3f} ({correct_l1/len(test_l3)})")
        print(f"    L1+L2 Correct: {hierarchical_l2_acc:.3f} ({correct_l2/len(test_l3)})")
        print(f"    L1+L2+L3 Correct: {hierarchical_full_acc:.3f} ({correct_full/len(test_l3)})")

        print("\n" + "="*70)

        return results
    
    def save(self, filepath):
        """
        Save the entire model to disk.
        """

        model_data = {
            'vectorizer_l1': self.vectorizer_l1,
            'vectorizer_l2': self.vectorizer_l2,
            'vectorizer_l3': self.vectorizer_l3,
            'model_l1': self.model_l1,
            'model_l2': self.model_l2,
            'model_l3': self.model_l3,
            'hierarchy_l1_to_l2': dict(self.hierarchy_l1_to_l2),
            'hierarchy_l2_to_l3': dict(self.hierarchy_l2_to_l3),
            'training_stats': self.training_stats,
            'text_preprocessor': self.text_preprocessor,
            'params': {
                'alpha_l1': self.alpha_l1,
                'alpha_l1': self.alpha_l1,
                'alpha_l1': self.alpha_l1,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df
            },
            'version': '2.0'
        }
        joblib.dump(model_data, filepath)
        print(f"\n Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from disk.
        """

        model_data = joblib.load(filepath)

        # Create instance
        obj = cls(
            alpha_l1 = model_data['params']['alpha_l1'],
            alpha_l2 = model_data['params']['alpha_l2'],
            alpha_l3 = model_data['params']['alpha_l3'],
            text_preprocessor=model_data.get('text_preprocessor', None)
        )

        obj.vectorizer_l1 = model_data['vectorizer_l1']
        obj.vectorizer_l2 = model_data['vectorizer_l2']
        obj.vectorizer_l3 = model_data['vectorizer_l3']
        obj.model_l1 = model_data['model_l1']
        obj.model_l2 = model_data['model_l2']
        obj.model_l3 = model_data['model_l3']
        obj.hierarchy_l1_to_l2 = defaultdict(set, {...})
        obj.hierarchy_l2_to_l3 = defaultdict(set, {...})
        obj.training_stats = model_data['training_stats']

        version = model_data.get('version', '1.0')

        return obj

