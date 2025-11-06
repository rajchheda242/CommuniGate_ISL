"""
Transformer-based model training for temporal ISL gesture sequences.
Implements a temporal transformer with attention mechanisms for sequence classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Configuration
DATA_DIR = "data/sequences_holistic"
MODEL_DIR = "models/transformer"
SEQUENCE_LENGTH = 90  # Match the actual data length
TARGET_LENGTH = 60    # Target length for training (will downsample)
FEATURE_DIM = 1662  # Holistic landmarks dimension
NUM_CLASSES = 5

PHRASES = [
    "Hi my name is Reet",
    "How are you", 
    "I am from Delhi",
    "I like coffee",
    "What do you like"
]

# Training hyperparameters (adjusted for smaller dataset)
BATCH_SIZE = 4  # Smaller batch size for limited data
LEARNING_RATE = 0.0001
EPOCHS = 50  # Reduced epochs for testing
PATIENCE = 10  # Reduced patience
DROPOUT = 0.1
NUM_HEADS = 4  # Reduced model complexity for small dataset
NUM_LAYERS = 3  # Reduced layers
D_MODEL = 256  # Smaller model
DFF = 1024  # Smaller feed forward


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TemporalTransformer(nn.Module):
    """Temporal Transformer for ISL gesture sequence classification."""
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, 
                 num_classes, dropout=0.1, max_seq_len=100):
        super(TemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=DFF,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head with confidence calibration
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        if return_attention:
            # For attention visualization (not implemented in this version)
            encoded = self.transformer(x)
            attention_weights = None
        else:
            encoded = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling across time dimension
        pooled = torch.mean(encoded, dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
        # Apply temperature scaling for calibration
        calibrated_logits = logits / self.temperature
        
        if return_attention:
            return calibrated_logits, attention_weights
        else:
            return calibrated_logits


class ISLDataset(Dataset):
    """Dataset class for ISL gesture sequences."""
    
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])


class TransformerTrainer:
    """Trainer class for Temporal Transformer model."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def resample_sequence(self, sequence, target_length):
        """Resample sequence to target length using interpolation."""
        from scipy import interpolate
        
        original_length = len(sequence)
        if original_length == target_length:
            return sequence
        
        # Create interpolation indices
        original_indices = np.linspace(0, original_length - 1, original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)
        
        # Interpolate each feature dimension
        resampled = np.zeros((target_length, sequence.shape[1]))
        for dim in range(sequence.shape[1]):
            f = interpolate.interp1d(original_indices, sequence[:, dim], kind='linear')
            resampled[:, dim] = f(target_indices)
        
        return resampled
    
    def load_data(self):
        """Load preprocessed holistic sequences."""
        print("Loading holistic sequence data...")
        
        X = []
        y = []
        loaded_count = 0
        
        for phrase_idx in range(len(PHRASES)):
            phrase_dir = os.path.join(DATA_DIR, f"phrase_{phrase_idx}")
            
            if not os.path.exists(phrase_dir):
                print(f"Warning: No data found for phrase {phrase_idx}")
                continue
            
            sequence_files = glob.glob(os.path.join(phrase_dir, "*_seq.npy"))
            phrase_loaded = 0
            
            for seq_file in sequence_files:
                try:
                    sequence = np.load(seq_file)
                    if sequence.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        # Resample from 90 frames to 60 frames for training
                        resampled = self.resample_sequence(sequence, TARGET_LENGTH)
                        X.append(resampled)
                        y.append(phrase_idx)
                        phrase_loaded += 1
                    else:
                        print(f"Warning: Invalid sequence shape in {seq_file}: {sequence.shape}")
                except Exception as e:
                    print(f"Error loading {seq_file}: {e}")
            
            loaded_count += phrase_loaded
            print(f"Phrase {phrase_idx}: Loaded {phrase_loaded} sequences - '{PHRASES[phrase_idx]}'")
        
        if len(X) == 0:
            raise ValueError(f"No valid sequence data found in {DATA_DIR}")
        
        print(f"Total loaded: {loaded_count} sequences")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal sequences loaded: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y
    
    def normalize_data(self, X_train, X_val, X_test):
        """Normalize sequence data."""
        print("Normalizing data...")
        
        # Reshape for normalization
        n_train, seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        # Fit scaler on training data
        X_train_norm = self.scaler.fit_transform(X_train_flat)
        X_val_norm = self.scaler.transform(X_val_flat)
        X_test_norm = self.scaler.transform(X_test_flat)
        
        # Reshape back
        X_train_norm = X_train_norm.reshape(n_train, seq_len, n_features)
        X_val_norm = X_val_norm.reshape(n_val, seq_len, n_features)
        X_test_norm = X_test_norm.reshape(n_test, seq_len, n_features)
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def create_model(self):
        """Create Temporal Transformer model."""
        self.model = TemporalTransformer(
            input_dim=FEATURE_DIM,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            max_seq_len=TARGET_LENGTH  # Use 60-frame target length
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel created:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, train_loader, val_loader):
        """Train the Transformer model."""
        print(f"\nStarting training...")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Patience: {PATIENCE}")
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\\n" + "="*70)
        print("Training Progress")
        print("="*70)
        
        for epoch in range(EPOCHS):
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print("  ✓ New best model saved")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\\nEarly stopping after {epoch+1} epochs")
                break
            
            print()
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Best model restored")
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set."""
        print("\\nEvaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(sequences)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Average Confidence: {np.mean(all_confidences):.4f}")
        
        # Classification report
        print("\\nClassification Report:")
        print(classification_report(
            all_labels, all_predictions, 
            target_names=PHRASES,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=PHRASES,
            yticklabels=PHRASES
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self):
        """Save trained model and metadata."""
        print("\\nSaving model...")
        
        # Save model state
        model_path = os.path.join(MODEL_DIR, "transformer_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': FEATURE_DIM,
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'num_classes': NUM_CLASSES,
                'dropout': DROPOUT,
                'sequence_length': SEQUENCE_LENGTH
            },
            'temperature': self.model.temperature.item()
        }, model_path)
        
        # Save scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save phrase mapping
        phrase_mapping = {i: phrase for i, phrase in enumerate(PHRASES)}
        mapping_path = os.path.join(MODEL_DIR, "phrase_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(phrase_mapping, f, indent=2)
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'TemporalTransformer',
            'sequence_length': SEQUENCE_LENGTH,
            'feature_dim': FEATURE_DIM,
            'num_classes': NUM_CLASSES,
            'phrases': PHRASES,
            'hyperparameters': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': len(self.train_losses),
                'dropout': DROPOUT,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'd_model': D_MODEL,
                'dff': DFF
            },
            'final_metrics': {
                'train_loss': self.train_losses[-1] if self.train_losses else 0,
                'val_loss': self.val_losses[-1] if self.val_losses else 0,
                'train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
                'val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0
            }
        }
        
        metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        print(f"✓ Phrase mapping saved to: {mapping_path}")
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def run(self):
        """Main training pipeline."""
        print("="*70)
        print("TEMPORAL TRANSFORMER TRAINING")
        print("="*70)
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\\nData split:")
        print(f"  Training: {len(X_train)} sequences")
        print(f"  Validation: {len(X_val)} sequences")
        print(f"  Test: {len(X_test)} sequences")
        
        # Normalize data
        X_train_norm, X_val_norm, X_test_norm = self.normalize_data(
            X_train, X_val, X_test
        )
        
        # Create datasets
        train_dataset = ISLDataset(X_train_norm, y_train)
        val_dataset = ISLDataset(X_val_norm, y_val)
        test_dataset = ISLDataset(X_test_norm, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        # Create model
        self.create_model()
        
        # Train model
        self.train_model(train_loader, val_loader)
        
        # Evaluate model
        test_accuracy = self.evaluate_model(test_loader)
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        self.save_model()
        
        print(f"\\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print("\\nModel saved to:", MODEL_DIR)
        print("\\nNext steps:")
        print("  Run: python inference.py")


def main():
    """Main training function."""
    trainer = TransformerTrainer()
    trainer.run()


if __name__ == "__main__":
    main()