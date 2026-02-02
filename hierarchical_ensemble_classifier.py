"""
Hierarchical Ensemble Classifier for Procurement Taxonomy Classification

This module implements a hierarchical ensemble approach where different ensemble
strategies are used at each level of the taxonomy:
- L1 (11 categories): Voting Ensemble (3 models)
- L2 (52 categories): Stacking Ensemble (4 base + 1 meta)
- L3 (135 categories): XGBoost (optimized for CPU)

Design Principles:
- Keep existing code intact (doesn't modify hierarchical_classifier.py)
- Modular architecture with clear separation of concerns
- Memory-efficient for 16GB RAM
- CPU-optimized for i7-1185G7
- Easy to swap models or strategies per level
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from preprocessing import TextPreprocessor, DataPreprocessor, normalize_category_name
import joblib
import xgboost as xgb
from collections import defaultdict
import warnings
import gc
import time
warnings.filterwarnings('ignore')


class EnsembleConfig:
    """
    Configuration class for ensemble parameters.
    Centralizes all hyperparameters for easy tuning.
    """
    
    def __init__(self):
        # TF-IDF parameters
        self.max_features = 10000  # Increased from 5000 for better accuracy
        self.ngram_range = (1, 2)  # Unigrams + bigrams
        self.min_df = 2  # Minimum document frequency
        
        # L1 parameters (Voting Ensemble)
        self.l1_alpha_nb = 0.05  # Lower alpha for confident predictions
        self.l1_alpha_cnb = 0.05
        
        # L2 parameters (Stacking Ensemble)
        self.l2_alpha_nb = 0.1
        self.l2_alpha_cnb = 0.1
        self.l2_cv_folds = 3  # For stacking meta-
        self.l2_use_calibrated_svc = True
        
        # L3 parameters (XGBoost)
        self.l3_n_estimators = 100
        self.l3_learning_rate = 0.1
        self.l3_max_depth = 4
        self.l3_subsample = 0.8
        self.l3_colsample_bytree = 0.8
        
        # GridSearch parameters
        self.use_gridsearch = True
        self.use_gridsearch_l1 = True
        self.use_gridsearch_l2 = False 
        self.use_gridsearch_l3 = True
        self.gridsearch_cv = 3
        self.gridsearch_n_jobs = -1  # Use all CPU cores
        
    def to_dict(self):
        """Export configuration as dictionary for saving."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_performance_estimate(self):
        """Estimate training time based on settings."""
        estimates = []
        if self.use_gridsearch and self.use_gridsearch_l1:
            estimates.append("L1: ~1-2 minutes")
        else:
            estimates.append("L1: ~30 seconds")

        if self.use_gridsearch and self.use_gridsearch_l2:
            estimates.append("L2: ~1-3 HOURS (nested CV)")
        else:
            estimates.append("L2: ~3-5 minutes")

        if self.use_gridsearch and self.use_gridsearch_l3:
            estimates.append("L3: ~3-10 minutes")
        else:
            estimates.append("L3: ~3-5 minutes")

        return estimates


class L1VotingEnsemble:
    """
    Level 1 Ensemble: Voting Classifier with 3 models.
    
    Strategy: Simple and fast for the easiest classification task.
    Models: MultinomialNB, ComplementNB, LogisticRegression
    """
    
    def __init__(self, config):
        self.config = config
        self.ensemble = None
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            lowercase=True,
            strip_accents=None
        )
        
    def build_ensemble(self):
        """Build the voting ensemble."""
        self.ensemble = VotingClassifier(
            estimators=[
                ('nb', MultinomialNB(alpha=self.config.l1_alpha_nb)),
                ('cnb', ComplementNB(alpha=self.config.l1_alpha_cnb)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ],
            voting='soft',  # Use probability scores
            n_jobs=-1  # Use all CPU cores
        )
            
    def train(self, train_df, description_col='description_clean', label_col='Επίπεδο Κατηγοριοποίησης L.1'):
        """Train the L1 ensemble."""
        print("\n[1/3] Training L1 Voting Ensemble (3 models)...")
        start_time = time.time()
        
        # Vectorize
        X_train = self.vectorizer.fit_transform(train_df[description_col])
        y_train = train_df[label_col]

        # Build base ensemble
        self.build_ensemble()

        # Check if GridSearch is enabled for L1
        use_gs = self.config.use_gridsearch and self.config.use_gridsearch_l1

        if use_gs:
            print("  [L1] Tuning hyperparameters with GridSearchCV...")
            # Parameter grid for VotingClassifier
            # Format: 'estimator_name__parameter_name'
            param_grid = {
                'nb__alpha': [0.01, 0.05, 0.1],
                'cnb__alpha': [0.01, 0.05, 0.1]
            }
            
            cv = StratifiedKFold(
                n_splits=self.config.gridsearch_cv,
                shuffle=True,
                random_state=42
            )
            
            grid_search = GridSearchCV(
                self.ensemble,
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=self.config.gridsearch_n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.ensemble = grid_search.best_estimator_
            
            print(f"  [L1] Best params: {grid_search.best_params_}")
            print(f"  [L1] Best CV score: {grid_search.best_score_:.3f}")
        else:
            print("  [L1] Training without GridSearch (faster)...")
            self.ensemble.fit(X_train, y_train)


        elapsed = time.time() - start_time
        print(f"  [L1] Training complete {elapsed:.1f}. Classes: {len(self.ensemble.classes_)}")
        
    def predict(self, text):
        """Predict L1 category and confidence."""
        X = self.vectorizer.transform([text])
        probs = self.ensemble.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        
        return {
            'prediction': self.ensemble.classes_[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {cls: float(prob) for cls, prob in zip(self.ensemble.classes_, probs)}
        }


class L2StackingEnsemble:
    """
    Level 2 Ensemble: Stacking Classifier with 4 base models + 1 meta-learner.
    
    Strategy: More sophisticated for moderate difficulty.
    Base models: MultinomialNB, ComplementNB, LogisticRegression, LinearSVC
    Meta-learner: LogisticRegression (learns how to combine base models)
    """
    
    def __init__(self, config):
        self.config = config
        self.ensemble = None
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            lowercase=True,
            strip_accents=None
        )
        
    def build_ensemble(self):
        """Build the stacking ensemble."""
        # Base models
        # We use CalibratedClassifierCV to add probability support
        if self.config.l2_use_calibrated_svc:
            svc = CalibratedClassifierCV(
                LinearSVC(class_weight='balanced', max_iter=2000, random_state=42),
                cv='prefit' #2, Minimal CV for calibration to save time
            )
        else:
            # Fallback: use LogisticRegression instead of SVC
            svc = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='saga',
                random_state=42
            )

        base_estimators = [
            ('nb', MultinomialNB(alpha=self.config.l2_alpha_nb)),
            ('cnb', ComplementNB(alpha=self.config.l2_alpha_cnb)),
            ('lr_lbfgs', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                penalty='l2',
                random_state=42
            )),
            ('lr_saga', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                penalty='l1',
                random_state=42
            ))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        # Stacking ensemble
        self.ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.config.l2_cv_folds,
            stack_method='predict_proba',  # Use probabilities
            n_jobs=-1
        )
            
    def train(self, train_df, description_col='description_clean', label_col='Επίπεδο Κατηγοριοποίησης L.2'):
        """Train the L2 ensemble."""
        print("\n[2/3] Training L2 Stacking Ensemble (4 base + 1 meta)...")
        start_time = time.time()

        # Vectorize
        X_train = self.vectorizer.fit_transform(train_df[description_col])
        y_train = train_df[label_col]
        
        # Validate data
        if y_train.isna().any():
            raise ValueError(f"L2 training data contains {y_train.isna().sum()} NaN values!")
        
        # Build base ensemble
        self.build_ensemble()

        # Check if GridSearch is enabled for L2
        use_gs = self.config.use_gridsearch and self.config.use_gridsearch_l2

        if use_gs:
            print("     [L2]: GridSearch enabled - this will take ~1-3 HOURS!")
            print("     [L2]: Set config.use_gridsearch_l2 = False for faster training")

            param_grid = {
                'nb__alpha': [0.05, 0.1, 0.15],
                'cnb__alpha': [0.05, 0.1, 0.15]
            }
            
            cv = StratifiedKFold(
                n_splits=self.config.gridsearch_cv,
                shuffle=True,
                random_state=42
            )
            
            grid_search = GridSearchCV(
                self.ensemble,
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=self.config.gridsearch_n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train) 
            self.ensemble = grid_search.best_estimator_
            
            print(f"  [L2] Best params: {grid_search.best_params_}")
            print(f"  [L2] Best CV score: {grid_search.best_score_:.3f}")
        else:
            print("  [L2] Training without GridSearch (recommended for speed)...")
            print(f"  [L2] Using {self.config.l2_cv_folds}-fold CV for stacking...")
            self.ensemble.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"  [L1] Training complete in {elapsed:.1f}. Classes: {len(self.ensemble.classes_)}")
        
    def predict(self, text):
        """Predict L2 category and confidence."""
        X = self.vectorizer.transform([text])
        probs = self.ensemble.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        
        return {
            'prediction': self.ensemble.classes_[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {cls: float(prob) for cls, prob in zip(self.ensemble.classes_, probs)}
        }


class L3XGBoostClassifier:
    """
    Level 3 Classifier: XGBoost optimized for CPU.
    
    Strategy: Powerful gradient boosting for the hardest classification task.
    Optimizations:
    - tree_method='hist': CPU-efficient
    - n_jobs=-1: Use all CPU cores
    - Moderate depth (4) to prevent overfitting
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            lowercase=True,
            strip_accents=None
        )
        
    def build_model(self, num_classes):
        """Build XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.l3_n_estimators,
            learning_rate=self.config.l3_learning_rate,
            max_depth=self.config.l3_max_depth,
            subsample=self.config.l3_subsample,
            colsample_bytree=self.config.l3_colsample_bytree,
            objective='multi:softprob',
            num_class=num_classes,
            tree_method='hist',  # CPU-optimized
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
            eval_metric='mlogloss'
        )
            
    def train(self, train_df, description_col='description_clean', label_col='Επίπεδο Κατηγοριοποίησης L.3'):
        """Train the L3 XGBoost model."""
        print("\n[3/3] Training L3 XGBoost Classifier (100 trees)...")
        start_time = time.time()
        
        # Vectorize
        X_train = self.vectorizer.fit_transform(train_df[description_col])
        y_train = train_df[label_col]
        
        # Validate data
        if y_train.isna().any():
            raise ValueError(f"L3 training data contains {y_train.isna().sum()} NaN values!")
        
        # Get number of classes
        num_classes = len(y_train.unique())

        # Build base model
        self.build_model(num_classes)

        # Check if GridSearch is enabled for L2
        use_gs = self.config.use_gridsearch and self.config.use_gridsearch_l3

        if use_gs:
            print("  [L3] Tuning hyperparameters with GridSearchCV...")
            param_grid = {
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
            
            cv = StratifiedKFold(
                n_splits=self.config.gridsearch_cv,
                shuffle=True,
                random_state=42
            )
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=self.config.gridsearch_n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            
            print(f"  [L3] Best params: {grid_search.best_params_}")
            print(f"  [L3] Best CV score: {grid_search.best_score_:.3f}")
        else:
            print("  [L3] Training without GridSearch (faster)...")
            self.model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"  [L1] Training complete {elapsed:.1f}. Classes: {num_classes}")
        
    def predict(self, text):
        """Predict L3 category and confidence."""
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        
        return {
            'prediction': self.model.classes_[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {cls: float(prob) for cls, prob in zip(self.model.classes_, probs)}
        }


class HierarchicalEnsembleClassifier:
    """
    Main Hierarchical Ensemble Classifier.
    
    Orchestrates the three-level ensemble hierarchy:
    1. L1: Voting Ensemble (fast and confident)
    2. L2: Stacking Ensemble (balanced and accurate)
    3. L3: XGBoost (powerful for complex task)
    
    Features:
    - Hierarchical filtering (L1 → L2 → L3)
    - Confidence-based predictions
    - Official taxonomy integration
    - Memory-efficient design
    - CPU-optimized
    """
    
    def __init__(self, config=None, text_preprocessor=None):
        """
        Initialize the hierarchical ensemble classifier.
        
        Parameters:
        -----------
        config : EnsembleConfig, optional
            Configuration object with hyperparameters
        text_preprocessor : TextPreprocessor, optional
            Text preprocessing object
        """
        self.config = config or EnsembleConfig()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        # Level-specific ensembles
        self.l1_ensemble = L1VotingEnsemble(self.config)
        self.l2_ensemble = L2StackingEnsemble(self.config)
        self.l3_classifier = L3XGBoostClassifier(self.config)
        
        # Hierarchy mappings
        self.hierarchy_l1_to_l2 = defaultdict(set)
        self.hierarchy_l2_to_l3 = defaultdict(set)
        
        # Official taxonomy
        self.official_hierarchy_l1_to_l2 = {}
        self.official_hierarchy_l2_to_l3 = {}
        
        # Training stats
        self.training_stats = {}

        # Print performance estimate
        print("\nTraining time estimates:")
        for est in self.config.get_performance_estimate():
            print(f"    {est}")
        
    def load_official_taxonomy(self, excel_path, sheet_name='L1-L2-L3'):
        """
        Load official taxonomy from Excel file.
        
        This ensures predictions follow the official hierarchy structure.
        """
        print(f"\n[2.5/3] Loading official taxonomy from '{sheet_name}' sheet...")
        
        # Read taxonomy sheet
        df_tax = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Detect column names
        l1_col = 'L1' if 'L1' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.1'
        l2_col = 'L2' if 'L2' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.2'
        l3_col = 'L3' if 'L3' in df_tax.columns else 'Επίπεδο Κατηγοριοποίησης L.3'
        
        # Handle merged cells (forward fill)
        df_tax[l1_col] = df_tax[l1_col].ffill()
        df_tax[l2_col] = df_tax[l2_col].ffill()
        df_tax = df_tax.dropna(subset=[l3_col])
        
        # Normalize category names
        df_tax[l1_col] = df_tax[l1_col].apply(normalize_category_name)
        df_tax[l2_col] = df_tax[l2_col].apply(normalize_category_name)
        df_tax[l3_col] = df_tax[l3_col].apply(normalize_category_name)
        
        # Build official hierarchy
        for _, row in df_tax.iterrows():
            l1, l2, l3 = row[l1_col], row[l2_col], row[l3_col]
            
            # L1 → L2 mapping
            if l1 not in self.official_hierarchy_l1_to_l2:
                self.official_hierarchy_l1_to_l2[l1] = set()
            self.official_hierarchy_l1_to_l2[l1].add(l2)
            
            # L2 → L3 mapping
            if l2 not in self.official_hierarchy_l2_to_l3:
                self.official_hierarchy_l2_to_l3[l2] = set()
            self.official_hierarchy_l2_to_l3[l2].add(l3)
        
        # Use official hierarchy for predictions
        self.hierarchy_l1_to_l2 = self.official_hierarchy_l1_to_l2
        self.hierarchy_l2_to_l3 = self.official_hierarchy_l2_to_l3
        
        print(f"  ✓ Official taxonomy loaded:")
        print(f"    L1 categories: {len(self.official_hierarchy_l1_to_l2)}")
        print(f"    L2 categories: {len(self.official_hierarchy_l2_to_l3)}")
        print(f"    L3 categories: {sum(len(v) for v in self.official_hierarchy_l2_to_l3.values())}")
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for training using DataPreprocessor.
        
        Returns datasets split by level (L1, L2, L3).
        """
        dataset_prep = DataPreprocessor(text_preprocessor=self.text_preprocessor)
        
        result = dataset_prep.prepare_dataset(
            df,
            test_size=test_size,
            random_state=random_state
        )
        
        # Extract prepared datasets
        train_l1, test_l1 = result['train_l1'], result['test_l1']
        train_l2, test_l2 = result['train_l2'], result['test_l2']
        train_l3, test_l3 = result['train_l3'], result['test_l3']
        
        # Store hierarchy from training data
        for l1, l2_set in result['hierarchy_l1_to_l2'].items():
            self.hierarchy_l1_to_l2[l1].update(l2_set)
        
        for l2, l3_set in result['hierarchy_l2_to_l3'].items():
            self.hierarchy_l2_to_l3[l2].update(l3_set)
        
        # Store preprocessing stats
        self.training_stats['preprocessing'] = result['stats']
        
        return (train_l1, test_l1), (train_l2, test_l2), (train_l3, test_l3)
    
    def train(self, train_l1, train_l2, train_l3):
        """
        Train all three levels of the ensemble hierarchy.
        
        Training order:
        1. L1 Voting Ensemble
        2. L2 Stacking Ensemble
        3. L3 XGBoost
        
        Each level is trained independently with its own data.
        """
        print("=" * 70)
        print("TRAINING HIERARCHICAL ENSEMBLE CLASSIFIER")
        print("=" * 70)

        total_start = time.time()
        
        # Train L1
        self.l1_ensemble.train(train_l1)
        gc.collect()  # Free memory
        
        # Train L2
        self.l2_ensemble.train(train_l2)
        gc.collect()  # Free memory
        
        # Train L3
        self.l3_classifier.train(train_l3)
        gc.collect()  # Free memory
        
        print("\n" + "=" * 70)
        print(f"ALL MODELS TRAINED in {time.time() - total_start:.1f}s")
        print("=" * 70)
        
    def predict(self, description, confidence_threshold=0.5):
        """
        Predict taxonomy for a procurement description.
        
        Process:
        1. Preprocess text
        2. Predict L1 with confidence
        3. Filter L2 options based on L1
        4. Predict L2 with confidence
        5. Filter L3 options based on L2
        6. Predict L3 with confidence
        7. Calculate combined confidence
        
        Returns:
        --------
        dict : Prediction results with confidence scores
        """
        # Preprocess text
        text_clean = self.text_preprocessor.preprocess(description)
        
        # ========== PREDICT L1 ==========
        l1_result = self.l1_ensemble.predict(text_clean)
        l1_pred = l1_result['prediction']
        l1_conf = l1_result['confidence']
        
        # ========== PREDICT L2 (filtered by L1) ==========
        valid_l2 = self.hierarchy_l1_to_l2.get(l1_pred, set())
        
        if len(valid_l2) == 0:
            return {
                'l1_pred': l1_pred,
                'l2_pred': None,
                'l3_pred': None,
                'l1_conf': l1_conf,
                'l2_conf': 0.0,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': 'No L2 children for predicted L1'
            }
        
        # Get L2 predictions
        l2_result = self.l2_ensemble.predict(text_clean)
        l2_probs = l2_result['probabilities']
        
        # Filter to valid L2 categories
        filtered_probs_l2 = {cls: prob for cls, prob in l2_probs.items() if cls in valid_l2}
        
        if not filtered_probs_l2:
            # No valid L2 found, use unfiltered prediction
            l2_pred = l2_result['prediction']
            l2_conf = l2_result['confidence']
        else:
            # Renormalize and select best
            total = sum(filtered_probs_l2.values())
            filtered_probs_l2 = {k: v/total for k, v in filtered_probs_l2.items()}
            l2_pred = max(filtered_probs_l2, key=filtered_probs_l2.get)
            l2_conf = filtered_probs_l2[l2_pred]
        
        # ========== PREDICT L3 (filtered by L2) ==========
        valid_l3 = self.hierarchy_l2_to_l3.get(l2_pred, set())
        
        if len(valid_l3) == 0:
            return {
                'l1_pred': l1_pred,
                'l2_pred': l2_pred,
                'l3_pred': None,
                'l1_conf': l1_conf,
                'l2_conf': l2_conf,
                'l3_conf': 0.0,
                'combined_conf': 0.0,
                'accept': False,
                'reason': 'No L3 children for predicted L2'
            }
        
        # Get L3 predictions
        l3_result = self.l3_classifier.predict(text_clean)
        l3_probs = l3_result['probabilities']
        
        # Filter to valid L3 categories
        filtered_probs_l3 = {cls: prob for cls, prob in l3_probs.items() if cls in valid_l3}
        
        if not filtered_probs_l3:
            # No valid L3 found, use unfiltered prediction
            l3_pred = l3_result['prediction']
            l3_conf = l3_result['confidence']
        else:
            # Renormalize and select best
            total = sum(filtered_probs_l3.values())
            filtered_probs_l3 = {k: v/total for k, v in filtered_probs_l3.items()}
            l3_pred = max(filtered_probs_l3, key=filtered_probs_l3.get)
            l3_conf = filtered_probs_l3[l3_pred]
        
        # ========== CALCULATE COMBINED CONFIDENCE ==========
        combined_conf = l1_conf * l2_conf * l3_conf
        
        return {
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
    
    def evaluate(self, test_l1, test_l2, test_l3):
        """
        Evaluate the ensemble on test data.
        
        Evaluates:
        1. Individual level performance (L1, L2, L3)
        2. Hierarchical pipeline performance
        
        Returns:
        --------
        dict : Evaluation metrics for each level
        """
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION")
        print("=" * 70)
        
        results = {}
        
        # ========== EVALUATE L1 ==========
        print("\n[1/3] Evaluating L1 Voting Ensemble...")
        X_test_l1 = self.l1_ensemble.vectorizer.transform(test_l1['description_clean'])
        y_test_l1 = test_l1['Επίπεδο Κατηγοριοποίησης L.1']
        y_pred_l1 = self.l1_ensemble.ensemble.predict(X_test_l1)
        
        l1_accuracy = (y_pred_l1 == y_test_l1).mean()
        l1_f1_weighted = f1_score(y_test_l1, y_pred_l1, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {l1_accuracy:.3f}")
        print(f"  F1-Weighted: {l1_f1_weighted:.3f}")
        
        results['l1'] = {
            'accuracy': l1_accuracy,
            'f1_weighted': l1_f1_weighted
        }
        
        # ========== EVALUATE L2 ==========
        print("\n[2/3] Evaluating L2 Stacking Ensemble...")
        X_test_l2 = self.l2_ensemble.vectorizer.transform(test_l2['description_clean'])
        y_test_l2 = test_l2['Επίπεδο Κατηγοριοποίησης L.2']
        y_pred_l2 = self.l2_ensemble.ensemble.predict(X_test_l2)
        
        l2_accuracy = (y_pred_l2 == y_test_l2).mean()
        l2_f1_weighted = f1_score(y_test_l2, y_pred_l2, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {l2_accuracy:.3f}")
        print(f"  F1-Weighted: {l2_f1_weighted:.3f}")
        
        results['l2'] = {
            'accuracy': l2_accuracy,
            'f1_weighted': l2_f1_weighted
        }
        
        # ========== EVALUATE L3 ==========
        print("\n[3/3] Evaluating L3 XGBoost Classifier...")
        X_test_l3 = self.l3_classifier.vectorizer.transform(test_l3['description_clean'])
        y_test_l3 = test_l3['Επίπεδο Κατηγοριοποίησης L.3']
        y_pred_l3 = self.l3_classifier.model.predict(X_test_l3)
        
        l3_accuracy = (y_pred_l3 == y_test_l3).mean()
        l3_f1_weighted = f1_score(y_test_l3, y_pred_l3, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {l3_accuracy:.3f}")
        print(f"  F1-Weighted: {l3_f1_weighted:.3f}")
        
        results['l3'] = {
            'accuracy': l3_accuracy,
            'f1_weighted': l3_f1_weighted
        }
        
        # ========== HIERARCHICAL EVALUATION ==========
        print("\n[HIERARCHICAL] Evaluating full pipeline...")
        
        correct_full = 0
        correct_l1 = 0
        correct_l2 = 0
        
        for idx, row in test_l3.iterrows():
            description = str(row.get('Συνοπτική Περιγραφή Αντικειμένου Σύμβασης', ''))
            
            try:
                pred = self.predict(description)
                
                actual_l1 = str(row['Επίπεδο Κατηγοριοποίησης L.1'])
                actual_l2 = str(row['Επίπεδο Κατηγοριοποίησης L.2'])
                actual_l3 = str(row['Επίπεδο Κατηγοριοποίησης L.3'])
                
                if pred['l1_pred'] == actual_l1:
                    correct_l1 += 1
                    if pred['l2_pred'] == actual_l2:
                        correct_l2 += 1
                        if pred['l3_pred'] == actual_l3:
                            correct_full += 1
            except:
                continue
        
        hierarchical_l1_acc = correct_l1 / len(test_l3)
        hierarchical_l2_acc = correct_l2 / len(test_l3)
        hierarchical_full_acc = correct_full / len(test_l3)
        
        results['hierarchical'] = {
            'l1_accuracy': hierarchical_l1_acc,
            'l2_accuracy': hierarchical_l2_acc,
            'full_accuracy': hierarchical_full_acc
        }
        
        # ========== SUMMARY ==========
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nIndividual Level Performance:")
        print(f"  L1 Accuracy: {l1_accuracy:.1%} | F1: {l1_f1_weighted:.3f}")
        print(f"  L2 Accuracy: {l2_accuracy:.1%} | F1: {l2_f1_weighted:.3f}")
        print(f"  L3 Accuracy: {l3_accuracy:.1%} | F1: {l3_f1_weighted:.3f}")
        print(f"\nHierarchical Pipeline Performance:")
        print(f"  L1 Correct: {hierarchical_l1_acc:.1%}")
        print(f"  L1+L2 Correct: {hierarchical_l2_acc:.1%}")
        print(f"  L1+L2+L3 Correct: {hierarchical_full_acc:.1%}")
        print("=" * 70)
        
        return results
    
    def save(self, filepath):
        """
        Save the entire ensemble to disk.
        
        Saves:
        - All three ensemble models
        - Vectorizers
        - Hierarchy mappings
        - Configuration
        - Training stats
        """
        model_data = {
            'l1_ensemble': {
                'ensemble': self.l1_ensemble.ensemble,
                'vectorizer': self.l1_ensemble.vectorizer
            },
            'l2_ensemble': {
                'ensemble': self.l2_ensemble.ensemble,
                'vectorizer': self.l2_ensemble.vectorizer
            },
            'l3_classifier': {
                'model': self.l3_classifier.model,
                'vectorizer': self.l3_classifier.vectorizer
            },
            'hierarchy_l1_to_l2': dict(self.hierarchy_l1_to_l2),
            'hierarchy_l2_to_l3': dict(self.hierarchy_l2_to_l3),
            'config': self.config.to_dict(),
            'training_stats': self.training_stats,
            'text_preprocessor': self.text_preprocessor,
            'version': '1.0_ensemble'
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Ensemble model saved to: {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """
        Load a trained ensemble from disk.
        
        Returns:
        --------
        HierarchicalEnsembleClassifier : Loaded classifier
        """
        print(f"Loading ensemble model from: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Recreate config
        config = EnsembleConfig()
        for key, value in model_data['config'].items():
            setattr(config, key, value)
        
        # Create instance
        obj = cls(
            config=config,
            text_preprocessor=model_data.get('text_preprocessor')
        )
        
        # Load L1 ensemble
        obj.l1_ensemble.ensemble = model_data['l1_ensemble']['ensemble']
        obj.l1_ensemble.vectorizer = model_data['l1_ensemble']['vectorizer']
        
        # Load L2 ensemble
        obj.l2_ensemble.ensemble = model_data['l2_ensemble']['ensemble']
        obj.l2_ensemble.vectorizer = model_data['l2_ensemble']['vectorizer']
        
        # Load L3 classifier
        obj.l3_classifier.model = model_data['l3_classifier']['model']
        obj.l3_classifier.vectorizer = model_data['l3_classifier']['vectorizer']
        
        # Load hierarchy
        obj.hierarchy_l1_to_l2 = defaultdict(set, model_data['hierarchy_l1_to_l2'])
        obj.hierarchy_l2_to_l3 = defaultdict(set, model_data['hierarchy_l2_to_l3'])
        
        # Load stats
        obj.training_stats = model_data['training_stats']
        
        print(f"✓ Ensemble model loaded successfully (version: {model_data.get('version')})")
        
        return obj
