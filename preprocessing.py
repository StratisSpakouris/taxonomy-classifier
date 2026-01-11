import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:

    def __init__(self, lowercase=True, normalize_spaces=True, keep_accents=True):
        self.lowercase = lowercase
        self.normaliz_spaces = normalize_spaces
        self.keep_accents = keep_accents

    def preprocess(self, text):
        if pd.isna(text):
            return ""
        
        # Convert to string in case of numeric input
        text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.normaliz_spaces:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        if not self.keep_accents:
            pass

        return text
    
    def validate(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return False, "Empty text"
        
        text = str(text).strip()

        if len(text) < 3:
            return False, f"Text too short ({len(text)} chars)"
        
        return True, ""
    

class DataPreprocessor:

    def __init__(self, text_preprocessor=None):

        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.hierarchy_l1_to_l2 = defaultdict(set)
        self.hierarchy_l1_to_l2 = defaultdict(set)
        self.preprocessing_report = {}

    def remove_duplicates(self, df):

        original_count = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = original_count - len(df_clean)

        stats = {
            'original_count': original_count,
            'duplicates_removed': duplicates_removed,
            'duplicates_percentage': duplicates_removed / original_count * 100 if original_count > 0 else 0,
            'final_clean_count': len(df_clean)
        }

        return df_clean, stats
    
    def clean_taxonomy_labels(self, df, label_columns):

        df_clean = df.copy()
        stats = {}

        for col in label_columns:
            if col not in df_clean.columns:
                continue

            # Track changes
            before = df_clean[col].copy()


            df_clean[col] = df_clean[col].astype(str)

            df_clean[col] = df_clean[col].replace('nan', pd.NA)

            # Strip whitespace
            df_clean[col] = df_clean[col].str.strip()

            # Remove newlines
            df_clean[col] = df_clean[col].replace('\n', ' ')
            df_clean[col] = df_clean[col].replace('\r', ' ')

            # Normalize multiple spaces
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)

            # Replace empty strings with NaN
            df_clean[col] = df_clean[col].replace('', pd.NA)

            # Track changes
            changes = (before != df_clean[col]).sum()
            stats[col] = {
                'changes': changes,
                'change_percentage': changes / len(df_clean) * 100 if len(df_clean) > 0 else 0
            }

        return df_clean, stats
    
    def build_hierarchy(self, df, l1_col, l2_col, l3_col):
        hierarchy_l1_to_l2 = defaultdict(set)
        hierarchy_l2_to_l3 = defaultdict(set)

        for l1, l2, in df[[l1_col, l2_col]].dropna().values:
            hierarchy_l1_to_l2[l1].add(l2)

        for l2, l3, in df[[l2_col, l3_col]].dropna().values:
            hierarchy_l2_to_l3[l2].add(l3)

        stats = {
            'l1_categories': len(hierarchy_l1_to_l2),
            'l2_categories': sum(len(v) for v in hierarchy_l1_to_l2.values()),
            'l3_categories': sum(len(v) for v in hierarchy_l2_to_l3.values())
        }

        # Store for later use
        self.hierarchy_l1_to_l2 = hierarchy_l1_to_l2
        self.hierarchy_l2_to_l3 = hierarchy_l2_to_l3

        return hierarchy_l1_to_l2, hierarchy_l2_to_l3, stats
    
    def handle_rare_categories(self, df, label_col, min_samples=2):
        value_counts = df[label_col].value_counts
        rare_categories = value_counts()[value_counts() < min_samples].index.tolist()

        if rare_categories:
            print(f"\nWarning: {len(rare_categories)} categories in '{label_col}' have < {min_samples} samples.")
            print("These will be excluded from training (insufficient data for stratified split).")

            df_filtered = df[~df[label_col].isin(rare_categories)].copy()
        else:
            df_filtered = df.copy()

        return df_filtered, rare_categories
    
    def create_train_test_split(self, df, label_col, test_size=0.2, random_state=42, stratify=True):

        if stratify:
            try:
                train, test = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=df[label_col]
                )
            except ValueError as e:
                # If stratification fails (e.g., categories with 1 sample)
                print(f"Stratification failed: {e}")
                print("Falling back to random split.")
                train, test = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state
                )
        else:
            train, test = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state
                )

        return train, test
    
    def prepare_dataset(self, df,
                        description_col='Συνοπτική Περιγραφή Αντικειμένου Σύμβασης',
                        l1_col='Επίπεδο Κατηγοριοποίησης L.1',
                        l2_col='Επίπεδο Κατηγοριοποίησης L.2',
                        l3_col='Επίπεδο Κατηγοριοποίησης L.3',
                        test_size=0.2,
                        random_state=42):
        
        print("="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)

        stats = {}

        # Step 1: Remove duplicates
        print("\nRemoving duplicates...")
        df_clean, dup_stats = self.remove_duplicates(df)
        stats['duplicates'] = dup_stats
        print(f"    Removed {dup_stats['duplicates_removed']} duplicates ({dup_stats['duplicates_percentage']:.1f}%)")

        # Step 2: Clean taxonomy labels
        print("\nCleaning taxonomy labels...")
        df_clean, clean_stats = self.clean_taxonomy_labels(
            df_clean,
            [l1_col, l2_col, l3_col]
        )
        stats['label_cleaning'] = clean_stats
        print(f"    Cleaned labels in {len(clean_stats)} columns")
        print(f"    Clean stats: {clean_stats}")

        # Step 3: Preprocess text descriptions
        print("\nPreprocessing text descriptions...")
        df_clean['description_clean'] = df_clean[description_col].apply(
            self.text_preprocessor.preprocess
        )
        print(f"    Preprocessed {len(df_clean)} descriptions")

        # Step 4: Prepare datesets for each level
        print("\nPreparing level-specific datasets...")

        # L1: Need at least L1 label
        df_l1 = df_clean.dropna(subset=[l1_col]).copy()
        print(f"    L1 dataset: {len(df_l1)} samples")

        # L2: Need both L1 and L2 labels
        df_l2 = df_clean.dropna(subset=[l1_col, l2_col]).copy()
        print(f"    L2 dataset: {len(df_l2)} samples")

        # L3: Need all three labels
        df_l3 = df_clean[
            (df_clean[l1_col].notna()) &
            (df_clean[l2_col].notna()) &
            (df_clean[l3_col].notna())
        ].copy()
        print(f"    L3 dataset: {len(df_l2)} samples")

        # Step 5: Build hierarchy
        print("\nBuilding taxonomy hierarchy...")
        hierarchy_l1_to_l2, hierarchy_l2_to_l3, hierarchy_stats = self.build_hierarchy(
            df_l1, l1_col, l2_col, l3_col
        )
        stats['hierarchy'] = hierarchy_stats
        print(f"    L1 categories: {hierarchy_stats['l1_categories']}")
        print(f"    L2 categories: {hierarchy_stats['l2_categories']}")
        print(f"    L3 categories: {hierarchy_stats['l3_categories']}")

        # Step 6: Handle rare categories
        print("\nHandling rare categories...")
        df_l1, rare_l1 = self. handle_rare_categories(df_l1, l1_col)
        df_l2, rare_l2 = self. handle_rare_categories(df_l2, l2_col)
        df_l3, rare_l3 = self. handle_rare_categories(df_l3, l3_col)

        stats['rare_categories'] = {
            'l1': rare_l1,
            'l2': rare_l2,
            'l3': rare_l3
        }

        # Step 7: Create train/test splits
        print(f"\nCreating train/test splits (test_size={test_size})...")

        train_l1, test_l1 = self.create_train_test_split(
            df_l1, l1_col, test_size, random_state
        )

        train_l2, test_l2 = self.create_train_test_split(
            df_l2, l2_col, test_size, random_state
        )

        train_l3, test_l3 = self.create_train_test_split(
            df_l3, l3_col, test_size, random_state
        )

        print(f"\n Train sizes: L1={len(train_l1)}, L2={len(train_l2)}, L3={len(train_l3)}")
        print(f"\n Test sizes: L1={len(test_l1)}, L2={len(test_l2)}, L3={len(test_l3)}")

        # Store statistics
        stats['split'] = {
            'train_l1': len(train_l1),
            'test_l1': len(test_l1),
            'train_l2': len(train_l2),
            'test_l2': len(test_l2),
            'train_l3': len(train_l3),
            'test_l3': len(test_l3),
        }

        self.preprocessing_report = stats

        print("="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)

        return {
            'train_l1': train_l1,
            'test_l1': test_l1,
            'train_l2': train_l2,
            'test_l2': test_l2,
            'train_l3': train_l3,
            'test_l3': test_l3,
            'hierarchy_l1_to_l2': dict(hierarchy_l1_to_l2),
            'hierarchy_l2_to_l3': dict(hierarchy_l2_to_l3),
            'stats': stats
        }

    def get_preprocessing_report(self):
        return self.preprocessing_report
    
def preprocess_procurement_data(df, test_size=0.2, random_state=42):
        
        preprocessor = DataPreprocessor()
        return preprocessor.prepare_dataset(df, test_size=test_size, random_state=random_state)