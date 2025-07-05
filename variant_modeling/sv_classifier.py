#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

class SVPathogenicityClassifier:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = None
        self.feature_importance = None
        
    def initialize_model(self, n_features=None):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000,
                C=1.0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def preprocess_features(self, X, y=None, fit=False):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            
            self.feature_selector = SelectKBest(
                score_func=f_classif, 
                k=min(50, X.shape[1])
            )
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
        else:
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
        
        return X_selected
    
    def handle_imbalanced_data(self, X, y):
        class_counts = np.bincount(y)
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            
            if imbalance_ratio > 3:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return X_resampled, y_resampled
        
        return X, y
    
    def train(self, X, y, validation_split=0.2, use_smote=True):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, 
            random_state=self.random_state, 
            stratify=y
        )
        
        X_train_processed = self.preprocess_features(X_train, y_train, fit=True)
        
        if use_smote:
            try:
                X_train_processed, y_train = self.handle_imbalanced_data(X_train_processed, y_train)
            except ImportError:
                print("SMOTE not available, proceeding without resampling")
        
        self.initialize_model()
        self.model.fit(X_train_processed, y_train)
        
        X_val_processed = self.preprocess_features(X_val)
        val_predictions = self.model.predict(X_val_processed)
        val_probabilities = self.model.predict_proba(X_val_processed)[:, 1]
        
        print("\nValidation Performance:")
        print(classification_report(y_val, val_predictions))
        print(f"ROC AUC: {roc_auc_score(y_val, val_probabilities):.3f}")
        
        self.calculate_feature_importance()
        
        return {
            'validation_accuracy': np.mean(val_predictions == y_val),
            'validation_auc': roc_auc_score(y_val, val_probabilities),
            'feature_importance': self.feature_importance
        }
    
    def cross_validate(self, X, y, cv_folds=5):
        X_processed = self.preprocess_features(X, y, fit=True)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        self.initialize_model()
        
        cv_scores = cross_val_score(self.model, X_processed, y, cv=cv, scoring='roc_auc')
        
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_processed = self.preprocess_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_processed = self.preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def calculate_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names) if self.feature_names else 0)
        
        selected_features = self.feature_selector.get_support()
        full_importances = np.zeros(len(selected_features))
        full_importances[selected_features] = importances
        
        if self.feature_names:
            self.feature_importance = dict(zip(self.feature_names, full_importances))
        else:
            self.feature_importance = {f"feature_{i}": imp for i, imp in enumerate(full_importances)}
    
    def get_top_features(self, n=20):
        if self.feature_importance is None:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:n])
    
    def plot_feature_importance(self, n=20, save_path=None):
        top_features = self.get_top_features(n)
        
        if not top_features:
            print("No feature importance available")
            return
        
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {n} Feature Importances - {self.model_type.title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Pathogenic'],
                    yticklabels=['Benign', 'Pathogenic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path=None):
        y_proba = self.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")
    
    def evaluate_model(self, X_test, y_test, detailed=True):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        accuracy = np.mean(y_pred == y_test)
        auc = roc_auc_score(y_test, y_proba)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        if detailed:
            print(f"\nTest Set Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"ROC AUC: {auc:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))
        
        return results

def create_synthetic_labels(features_df, pathogenic_fraction=0.3):
    np.random.seed(42)
    n_samples = len(features_df)
    n_pathogenic = int(n_samples * pathogenic_fraction)
    
    labels = np.zeros(n_samples)
    labels[:n_pathogenic] = 1
    np.random.shuffle(labels)
    
    high_impact_mask = (
        (features_df.get('in_exon', 0) == 1) |
        (features_df.get('log_sv_length', 0) > 3) |
        (features_df.get('is_rare', 0) == 1)
    )
    
    labels[high_impact_mask] = np.random.choice([0, 1], size=np.sum(high_impact_mask), p=[0.3, 0.7])
    
    return labels.astype(int)

def main():
    parser = argparse.ArgumentParser(description='Train SV pathogenicity classifier')
    parser.add_argument('--features', required=True, help='Features file (TSV)')
    parser.add_argument('--labels', help='Labels file (TSV with pathogenic column)')
    parser.add_argument('--model-type', choices=['random_forest', 'gradient_boosting', 'logistic'], 
                       default='random_forest', help='Model type')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    features_df = pd.read_csv(args.features, sep='\t')
    
    if args.labels:
        labels_df = pd.read_csv(args.labels, sep='\t')
        y = labels_df['pathogenic'].values
    else:
        print("No labels provided, creating synthetic labels based on features")
        y = create_synthetic_labels(features_df)
    
    print(f"Loaded {len(features_df)} variants with {len(features_df.columns)} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    classifier = SVPathogenicityClassifier(model_type=args.model_type)
    
    if args.cross_validate:
        cv_scores = classifier.cross_validate(features_df, y)
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_auc': float(cv_scores.mean()),
            'std_cv_auc': float(cv_scores.std())
        }
        
        with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    classifier.plot_feature_importance(save_path=output_dir / 'feature_importance.png')
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    test_results = classifier.evaluate_model(X_test, y_test)
    classifier.plot_confusion_matrix(X_test, y_test, save_path=output_dir / 'confusion_matrix.png')
    classifier.plot_precision_recall_curve(X_test, y_test, save_path=output_dir / 'precision_recall.png')
    
    print(f"\nResults saved to {output_dir}")
    print(f"Model performance: AUC = {test_results['auc']:.3f}")

if __name__ == "__main__":
    main()cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
    
    training_results = classifier.train(features_df, y)
    
    classifier.save_model(output_dir / 'sv_classifier_model.pkl')
    
    with open(output_dir / '