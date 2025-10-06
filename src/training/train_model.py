"""
Model training script for ISL gesture recognition.
Trains a classifier on collected landmark data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import glob


DATA_DIR = "data/processed"
MODEL_DIR = "models/saved"


class GestureModelTrainer:
    """Trains and evaluates gesture recognition models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def load_data(self):
        """Load all CSV files from data directory."""
        csv_files = glob.glob(os.path.join(DATA_DIR, "gesture_data_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No data files found in {DATA_DIR}")
        
        print(f"Found {len(csv_files)} data file(s)")
        
        # Load and combine all CSV files
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded {file}: {len(df)} samples")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal samples: {len(combined_df)}")
        
        # Extract features (landmarks) and labels (phrase_id)
        feature_cols = [col for col in combined_df.columns if col.startswith('landmark_')]
        X = combined_df[feature_cols].values
        y = combined_df['phrase_id'].values
        
        print(f"Feature shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y, combined_df
    
    def train_knn(self, X_train, y_train):
        """Train K-Nearest Neighbors classifier."""
        print("\nTraining KNN classifier...")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        return model
    
    def train_svm(self, X_train, y_train):
        """Train Support Vector Machine classifier."""
        print("\nTraining SVM classifier...")
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        print(f"\n{'='*60}")
        print(f"{model_name} Evaluation")
        print(f"{'='*60}")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model, scaler, model_name):
        """Save trained model and scaler."""
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def train(self):
        """Main training pipeline."""
        print("ISL Gesture Model Training")
        print("="*60)
        
        # Load data
        X, y, df = self.load_data()
        
        # Check class distribution
        print("\nClass distribution:")
        for phrase_id in np.unique(y):
            count = np.sum(y == phrase_id)
            phrase_text = df[df['phrase_id'] == phrase_id]['phrase_text'].iloc[0]
            print(f"  Phrase {phrase_id}: {count} samples - '{phrase_text}'")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate both models
        knn_model = self.train_knn(X_train_scaled, y_train)
        knn_accuracy = self.evaluate_model(knn_model, X_test_scaled, y_test, "KNN")
        
        svm_model = self.train_svm(X_train_scaled, y_train)
        svm_accuracy = self.evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
        
        # Save the best model
        if svm_accuracy >= knn_accuracy:
            best_model = svm_model
            best_name = "svm"
            print(f"\nSVM selected as best model (Accuracy: {svm_accuracy:.2%})")
        else:
            best_model = knn_model
            best_name = "knn"
            print(f"\nKNN selected as best model (Accuracy: {knn_accuracy:.2%})")
        
        self.save_model(best_model, self.scaler, best_name)
        
        # Save phrase mapping
        phrase_mapping = {}
        for phrase_id in np.unique(y):
            phrase_text = df[df['phrase_id'] == phrase_id]['phrase_text'].iloc[0]
            phrase_mapping[int(phrase_id)] = phrase_text
        
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.joblib")
        joblib.dump(phrase_mapping, mapping_path)
        print(f"Phrase mapping saved to: {mapping_path}")
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)


if __name__ == "__main__":
    trainer = GestureModelTrainer()
    trainer.train()
