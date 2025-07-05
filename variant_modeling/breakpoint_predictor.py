#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import pickle

class BreakpointPredictor:
    def __init__(self, sequence_length=1000, model_type='cnn_lstm'):
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.nucleotide_encoder = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.history = None
        
    def encode_sequence(self, sequence):
        if len(sequence) == 0:
            return np.zeros(self.sequence_length)
        
        sequence = sequence.upper()
        encoded = [self.nucleotide_encoder.get(base, 4) for base in sequence]
        
        if len(encoded) > self.sequence_length:
            encoded = encoded[:self.sequence_length]
        else:
            encoded.extend([4] * (self.sequence_length - len(encoded)))
        
        return np.array(encoded)
    
    def one_hot_encode_sequence(self, sequence):
        encoded = self.encode_sequence(sequence)
        one_hot = np.zeros((self.sequence_length, 5))
        
        for i, nucleotide in enumerate(encoded):
            if nucleotide < 5:
                one_hot[i, int(nucleotide)] = 1
        
        return one_hot
    
    def prepare_sequence_data(self, sequences):
        if self.model_type in ['cnn', 'cnn_lstm', 'attention']:
            return np.array([self.one_hot_encode_sequence(seq) for seq in sequences])
        else:
            return np.array([self.encode_sequence(seq) for seq in sequences])
    
    def create_cnn_model(self, input_shape):
        model = Sequential([
            Conv1D(64, kernel_size=10, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=3),
            Conv1D(128, kernel_size=8, activation='relu'),
            MaxPooling1D(pool_size=3),
            Conv1D(256, kernel_size=6, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        return model
    
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model
    
    def create_cnn_lstm_model(self, input_shape):
        model = Sequential([
            Conv1D(64, kernel_size=10, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=8, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model
    
    def create_attention_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        conv1 = Conv1D(64, kernel_size=10, activation='relu')(inputs)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        
        conv2 = Conv1D(128, kernel_size=8, activation='relu')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        
        lstm = LSTM(128, return_sequences=True)(pool2)
        
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=16
        )(lstm, lstm)
        
        global_avg = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        dense1 = Dense(256, activation='relu')(global_avg)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dropout1)
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model(self, input_shape):
        if self.model_type == 'cnn':
            self.model = self.create_cnn_model(input_shape)
        elif self.model_type == 'lstm':
            self.model = self.create_lstm_model(input_shape)
        elif self.model_type == 'cnn_lstm':
            self.model = self.create_cnn_lstm_model(input_shape)
        elif self.model_type == 'attention':
            self.model = self.create_attention_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_training_data(self, sequences, breakpoint_positions, read_positions):
        X_seq = self.prepare_sequence_data(sequences)
        
        numerical_features = np.column_stack([
            read_positions,
            [len(seq) for seq in sequences],
            [seq.count('N') / max(1, len(seq)) for seq in sequences]
        ])
        
        X_num = self.scaler.fit_transform(numerical_features)
        
        y = np.array(breakpoint_positions)
        
        return X_seq, X_num, y
    
    def train(self, sequences, breakpoint_positions, read_positions, 
              validation_split=0.2, epochs=100, batch_size=32):
        
        X_seq, X_num, y = self.prepare_training_data(sequences, breakpoint_positions, read_positions)
        
        X_train_seq, X_val_seq, X_train_num, X_val_num, y_train, y_val = train_test_split(
            X_seq, X_num, y, test_size=validation_split, random_state=42
        )
        
        if self.model_type in ['cnn', 'cnn_lstm', 'attention']:
            input_shape = (self.sequence_length, 5)
        else:
            input_shape = (self.sequence_length, 1)
            X_train_seq = X_train_seq.reshape(-1, self.sequence_length, 1)
            X_val_seq = X_val_seq.reshape(-1, self.sequence_length, 1)
        
        self.build_model(input_shape)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7),
            ModelCheckpoint('best_breakpoint_model.h5', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        val_predictions = self.model.predict(X_val_seq)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        print(f"\nValidation Performance:")
        print(f"MAE: {val_mae:.2f} bp")
        print(f"RMSE: {val_rmse:.2f} bp")
        
        return {
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'history': self.history.history
        }
    
    def predict(self, sequences, read_positions):
        X_seq = self.prepare_sequence_data(sequences)
        
        numerical_features = np.column_stack([
            read_positions,
            [len(seq) for seq in sequences],
            [seq.count('N') / max(1, len(seq)) for seq in sequences]
        ])
        
        X_num = self.scaler.transform(numerical_features)
        
        if self.model_type not in ['cnn', 'cnn_lstm', 'attention']:
            X_seq = X_seq.reshape(-1, self.sequence_length, 1)
        
        predictions = self.model.predict(X_seq)
        return predictions.flatten()
    
    def plot_training_history(self, save_path=None):
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_vs_actual(self, sequences, actual_positions, read_positions, save_path=None):
        predictions = self.predict(sequences, read_positions)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(actual_positions, predictions, alpha=0.6)
        
        min_val = min(min(actual_positions), min(predictions))
        max_val = max(max(actual_positions), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('Actual Breakpoint Position')
        plt.ylabel('Predicted Breakpoint Position')
        plt.title('Breakpoint Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        mae = mean_absolute_error(actual_positions, predictions)
        rmse = np.sqrt(mean_squared_error(actual_positions, predictions))
        plt.text(0.05, 0.95, f'MAE: {mae:.1f} bp\nRMSE: {rmse:.1f} bp', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        model_dir = Path(filepath).parent
        model_dir.mkdir(exist_ok=True)
        
        self.model.save(f"{filepath}_model.h5")
        
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata = {
            'sequence_length': self.sequence_length,
            'model_type': self.model_type,
            'nucleotide_encoder': self.nucleotide_encoder
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.sequence_length = metadata['sequence_length']
        self.model_type = metadata['model_type']
        self.nucleotide_encoder = metadata['nucleotide_encoder']
        
        print(f"Model loaded from {filepath}")

def generate_synthetic_data(n_samples=1000, sequence_length=1000):
    sequences = []
    breakpoint_positions = []
    read_positions = []
    
    np.random.seed(42)
    
    for i in range(n_samples):
        sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=sequence_length))
        
        true_breakpoint = np.random.randint(100, sequence_length - 100)
        
        if np.random.random() < 0.3:
            insertion_seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=np.random.randint(10, 50)))
            sequence = sequence[:true_breakpoint] + insertion_seq + sequence[true_breakpoint:]
        
        sequences.append(sequence)
        breakpoint_positions.append(true_breakpoint)
        read_positions.append(np.random.randint(0, 10000))
    
    return sequences, breakpoint_positions, read_positions

def main():
    parser = argparse.ArgumentParser(description='Train breakpoint prediction model')
    parser.add_argument('--sequences', help='Input sequences file (FASTA or text)')
    parser.add_argument('--breakpoints', help='Breakpoint positions file (TSV)')
    parser.add_argument('--model-type', choices=['cnn', 'lstm', 'cnn_lstm', 'attention'], 
                       default='cnn_lstm', help='Model architecture')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--sequence-length', type=int, default=1000, help='Sequence length')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.synthetic or not args.sequences:
        print("Generating synthetic training data...")
        sequences, breakpoint_positions, read_positions = generate_synthetic_data(
            n_samples=2000, sequence_length=args.sequence_length
        )
    else:
        print("Loading real data not implemented - using synthetic data")
        sequences, breakpoint_positions, read_positions = generate_synthetic_data(
            n_samples=2000, sequence_length=args.sequence_length
        )
    
    print(f"Training on {len(sequences)} sequences")
    
    predictor = BreakpointPredictor(
        sequence_length=args.sequence_length,
        model_type=args.model_type
    )
    
    training_results = predictor.train(
        sequences, breakpoint_positions, read_positions,
        epochs=args.epochs
    )
    
    predictor.save_model(output_dir / 'breakpoint_predictor')
    
    with open(output_dir / 'training_results.json', 'w') as f:
        training_results_serializable = {
            'val_mae': float(training_results['val_mae']),
            'val_rmse': float(training_results['val_rmse'])
        }
        json.dump(training_results_serializable, f, indent=2)
    
    predictor.plot_training_history(save_path=output_dir / 'training_history.png')
    
    test_sequences = sequences[-200:]
    test_positions = breakpoint_positions[-200:]
    test_read_pos = read_positions[-200:]
    
    predictor.plot_predictions_vs_actual(
        test_sequences, test_positions, test_read_pos,
        save_path=output_dir / 'predictions_vs_actual.png'
    )
    
    print(f"\nResults saved to {output_dir}")
    print(f"Model performance: MAE = {training_results['val_mae']:.1f} bp")

if __name__ == "__main__":
    main()