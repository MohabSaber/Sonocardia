import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import psutil
import noisereduce as nr
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import gc
import warnings
import scipy.stats
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
import seaborn as sns
warnings.filterwarnings('ignore')

class UnifiedHeartSoundClassifier:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            tree_method='hist',
            enable_categorical=False,
            eval_metric='mlogloss'
        )
        self.nn_model = None
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.feature_stats = {}
        
    def validate_audio_file(self, file_path):
        """Validate if the audio file is readable and has content."""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            
            if os.path.getsize(file_path) == 0:
                print(f"Empty file: {file_path}")
                return False
                
            return True
        except Exception as e:
            print(f"Error validating file {file_path}: {str(e)}")
            return False

    def ml_noise_reduction(self, y, sr, noise_duration=0.5):
        try:
            if len(y) < int(noise_duration * sr):
                print("Warning: Audio signal too short for noise reduction")
                return y
            noise_sample = y[:int(noise_duration * sr)]
            y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
            return y_denoised
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return y

    def bandpass_filter(self, y, sr, low_cut=20, high_cut=500):
        try:
            nyquist = 0.5 * sr
            low = low_cut / nyquist
            high = high_cut / nyquist
            if high > 1.0:
                high = 0.99
            b, a = butter(1, [low, high], btype='band')
            filtered = filtfilt(b, a, y)
            if np.all(np.isnan(filtered)):
                print("Warning: Filtering produced NaN values")
                return y
            return filtered
        except Exception as e:
            print(f"Bandpass filter error: {e}")
            return y

    def preprocess_audio(self, file_path, target_sr=22050, noise_duration=0.5):
        try:
            if not self.validate_audio_file(file_path):
                return None
                
            try:
                y, sr = librosa.load(file_path, sr=target_sr)
            except Exception as e:
                print(f"Error loading audio file {file_path}: {str(e)}")
                return None

            if np.all(y == 0) or np.max(np.abs(y)) < 1e-6:
                print(f"Warning: Silent or near-silent audio in {file_path}")
                return None

            y = self.ml_noise_reduction(y, sr, noise_duration)
            if y is None or len(y) == 0:
                print(f"Error: Noise reduction failed for {file_path}")
                return None

            y = self.bandpass_filter(y, sr)
            if y is None or len(y) == 0:
                print(f"Error: Bandpass filtering failed for {file_path}")
                return None

            y, _ = librosa.effects.trim(y, top_db=20)
            if len(y) == 0:
                print(f"Warning: Empty signal after trimming for {file_path}")
                return None

            max_abs = np.max(np.abs(y))
            if max_abs > 0:
                y = y / max_abs
            else:
                print(f"Warning: Zero signal amplitude in {file_path}")
                return None

            return y

        except Exception as e:
            print(f"Error preprocessing file {file_path}: {str(e)}")
            return None
    def extract_features(self, signal, sr=22050):
        """
        A comprehensive feature extraction function without harmonic features.
        """
        try:
            # Input validation and signal preparation
            if signal is None or len(signal) == 0:
                print("Invalid input signal")
                return None

            print(f"Processing signal: length={len(signal)}, sr={sr}")
            
            # Normalize signal
            signal = signal / (np.max(np.abs(signal)) + 1e-10)
            
            # Dictionary to store all feature sets
            feature_sets = {}
            
            # 1. Time Domain Features
            try:
                print("Extracting time domain features...")
                
                rms = librosa.feature.rms(y=signal)[0]
                zcr = librosa.feature.zero_crossing_rate(signal)[0]
                mean = np.mean(signal)
                std = np.std(signal)
                skew = scipy.stats.skew(signal)
                kurtosis = scipy.stats.kurtosis(signal)
                
                time_features = np.array([
                    np.mean(rms),
                    np.std(rms),
                    np.mean(zcr),
                    np.std(zcr),
                    mean,
                    std,
                    skew,
                    kurtosis
                ])
                
                feature_sets['time_domain'] = time_features
                print("Time domain features extracted successfully")
                
            except Exception as e:
                print(f"Time domain feature extraction failed: {e}")
                return None

            # 2. STFT-based Features
            try:
                print("Extracting STFT-based features...")
                
                D = librosa.stft(signal, n_fft=2048, hop_length=512)
                magnitude_spectrum = np.abs(D)
                
                centroid = librosa.feature.spectral_centroid(S=magnitude_spectrum)[0]
                rolloff = librosa.feature.spectral_rolloff(S=magnitude_spectrum)[0]
                bandwidth = librosa.feature.spectral_bandwidth(S=magnitude_spectrum)[0]
                contrast = librosa.feature.spectral_contrast(S=magnitude_spectrum)[0]
                
                freq_features = np.concatenate([
                    [np.mean(centroid), np.std(centroid)],
                    [np.mean(rolloff), np.std(rolloff)],
                    [np.mean(bandwidth), np.std(bandwidth)],
                    [np.mean(contrast), np.std(contrast)]
                ])
                
                feature_sets['freq_domain'] = freq_features
                print("STFT-based features extracted successfully")
                
            except Exception as e:
                print(f"STFT-based feature extraction failed: {e}")
                return None

            # 3. Mel-based Features
            try:
                print("Extracting Mel-based features...")
                
                mel_spec = librosa.feature.melspectrogram(
                    y=signal,
                    sr=sr,
                    n_mels=128,
                    n_fft=2048,
                    hop_length=512
                )
                
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=20)
                mfcc_delta = librosa.feature.delta(mfcc)
                
                mel_features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    np.std(mfcc, axis=1),
                    np.mean(mfcc_delta, axis=1)
                ])
                
                feature_sets['mel_based'] = mel_features
                print("Mel-based features extracted successfully")
                
            except Exception as e:
                print(f"Mel-based feature extraction failed: {e}")
                return None

            # 4. Rhythm Features
            try:
                print("Extracting rhythm features...")
                
                tempo, beats = librosa.beat.beat_track(y=signal, sr=sr)
                beat_frames = librosa.frames_to_time(beats, sr=sr)
                
                rhythm_features = np.array([
                    tempo,
                    np.mean(np.diff(beat_frames)) if len(beat_frames) > 1 else 0,
                    np.std(np.diff(beat_frames)) if len(beat_frames) > 1 else 0,
                    len(beats) / (len(signal) / sr)  # Beat density
                ])
                
                feature_sets['rhythm'] = rhythm_features
                print("Rhythm features extracted successfully")
                
            except Exception as e:
                print(f"Rhythm feature extraction failed: {e}")
                return None

            # Combine all features
            try:
                print("Combining features...")
                final_features = np.concatenate([
                    feature_sets['time_domain'],
                    feature_sets['freq_domain'],
                    feature_sets['mel_based'],
                    feature_sets['rhythm']
                ])
                
                print(f"Final feature vector shape: {final_features.shape}")
                return final_features
                
            except Exception as e:
                print(f"Error combining features: {e}")
                return None

        except Exception as e:
            print(f"Unexpected error in feature extraction: {e}")
            return None
    def evaluate_feature_combinations(self, X, y):
        """Evaluate different feature combinations to find the best performing set."""
        feature_start_indices = {}
        current_idx = 0
        
        # Build feature indices based on feature_stats
        for feature_name, feature_dim in self.feature_stats.items():
            feature_start_indices[feature_name] = current_idx
            current_idx += feature_dim
        
        # Define combinations to test
        combinations = [
            ['time_domain'],
            ['freq_domain_1'],
            ['mel_based'],
            ['time_domain', 'freq_domain_1'],
            ['time_domain', 'mel_based'],
            ['freq_domain_1', 'mel_based'],
            ['time_domain', 'freq_domain_1', 'mel_based'],
            ['rhythm', 'harmonic'],
            ['time_domain', 'freq_domain_1', 'mel_based', 'rhythm'],
            ['time_domain', 'freq_domain_1', 'mel_based', 'harmonic'],
            ['time_domain', 'freq_domain_1', 'mel_based', 'rhythm', 'harmonic']
        ]
        
        results = {}
        for combo in combinations:
            print(f"\nTesting combination: {combo}")
            indices = []
            for feature_set in combo:
                if feature_set in feature_start_indices:
                    start_idx = feature_start_indices[feature_set]
                    end_idx = start_idx + self.feature_stats[feature_set]
                    indices.extend(range(start_idx, end_idx))
                else:
                    print(f"Warning: Feature set {feature_set} not found in extracted features")
            
            if indices:
                X_subset = X[:, indices]
                accuracy = self.train_and_evaluate_subset(X_subset, y)
                results[str(combo)] = accuracy
            
        return results
    def prepare_data(self, data_dir, labels):
        """Prepare and extract features from audio files."""
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        all_files = []
        processed_features = []
        valid_labels = []
        
        # Initialize progress tracking
        total_processed = 0
        failed_files = []

        # Gather all files
        for label in labels:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory not found - {label_dir}")
                continue
                
            files = [f for f in os.listdir(label_dir) 
                    if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            
            if not files:
                print(f"Warning: No audio files found in {label_dir}")
                continue
                
            all_files.extend([[os.path.join(label_dir, file), label] for file in files])

        if not all_files:
            raise ValueError("No valid audio files found in the data directory")

        data_df = pd.DataFrame(all_files, columns=['file_path', 'label'])
        data_df['label'] = pd.Categorical(data_df['label']).codes

        total_files = len(data_df)
        print(f"Found {total_files} files to process")

        # Process each file
        for idx, (_, row) in enumerate(data_df.iterrows(), 1):
            try:
                print(f"\nProcessing file {idx}/{total_files}: {row['file_path']}", flush=True)
                
                # Memory management
                if idx % 10 == 0:
                    gc.collect()
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    print(f"Current memory usage: {current_memory:.2f} MB")
                
                # Preprocess audio
                signal = self.preprocess_audio(row['file_path'])
                if signal is None:
                    print(f"Skipping file due to preprocessing failure: {row['file_path']}")
                    failed_files.append((row['file_path'], "preprocessing failure"))
                    continue

                # Extract features
                features = self.extract_features(signal)
                if features is None:
                    print(f"Skipping file due to feature extraction failure: {row['file_path']}")
                    failed_files.append((row['file_path'], "feature extraction failure"))
                    continue

                processed_features.append(features)
                valid_labels.append(row['label'])
                total_processed += 1

                # Progress update
                if idx % 10 == 0:
                    print(f"Successfully processed {total_processed}/{total_files} files ({(total_processed/total_files)*100:.1f}%)")
                
                # Memory cleanup
                del signal
                del features
                if idx % 25 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error processing file {row['file_path']}: {str(e)}")
                failed_files.append((row['file_path'], str(e)))
                continue

        if not processed_features:
            raise ValueError("No features were successfully extracted from any files")

        # Print processing summary
        print("\nProcessing Summary:")
        print(f"Total files attempted: {total_files}")
        print(f"Successfully processed: {total_processed}")
        print(f"Failed files: {len(failed_files)}")
        
        if failed_files:
            print("\nFailed files and reasons:")
            for file, reason in failed_files[:10]:  # Show first 10 failed files
                print(f"- {file}: {reason}")
            if len(failed_files) > 10:
                print(f"... and {len(failed_files) - 10} more")

        X = np.array(processed_features)
        y = np.array(valid_labels)

        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Data validation
        if np.any(np.isnan(X)):
            raise ValueError("Dataset contains NaN values")
        if len(X) != len(y):
            raise ValueError("Feature and label arrays have different lengths")

        X = self.scaler.fit_transform(X)
        return X, y

    def build_nn_model(self, input_shape, num_classes):
        """Build and compile the neural network model."""
        self.nn_model = Sequential([
            Input(shape=input_shape),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.nn_model

    def train_and_evaluate_subset(self, X, y):
        """Train and evaluate models on a subset of features."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        
        return xgb_accuracy

    def train_and_evaluate(self, X, y):
        """Train and evaluate all models."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = {}

        # Train XGBoost
        print("\nTraining XGBoost...")
        self.xgb_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=True
        )
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = xgb_accuracy

        # Train Neural Network
        print("\nTraining Neural Network...")
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
        
        self.build_nn_model((X_train.shape[1],), len(np.unique(y)))
        
        history = self.nn_model.fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        nn_pred = np.argmax(self.nn_model.predict(X_test), axis=1)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        results['Neural Network'] = nn_accuracy

        # Train Ensemble
        print("\nTraining Ensemble Model...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('nn', LogisticRegression(multi_class='ovr', max_iter=1000))
            ],
            voting='soft'
        )
        self.ensemble_model.fit(X_train, y_train)
        ensemble_pred = self.ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        results['Ensemble'] = ensemble_accuracy

        # Visualizations
        self.plot_results(results)
        self.plot_confusion_matrices(y_test, xgb_pred, nn_pred, ensemble_pred)
        self.plot_learning_curves(history)

        return results

    def plot_results(self, results):
        """Plot comparison of model accuracies."""
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Model Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, v in enumerate(results.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.show()

    def plot_confusion_matrices(self, y_true, xgb_pred, nn_pred, ensemble_pred):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        predictions = [xgb_pred, nn_pred, ensemble_pred]
        titles = ['XGBoost', 'Neural Network', 'Ensemble']
        
        for ax, pred, title in zip(axes, predictions, titles):
            cm = confusion_matrix(y_true, pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'{title} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, history):
        """Plot neural network learning curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    def save_model(self, filepath):
        """Save all models and preprocessing components."""
        try:
            model_data = {
                'xgb_model': self.xgb_model,
                'nn_model': self.nn_model,
                'scaler': self.scaler,
                'ensemble_model': self.ensemble_model,
                'feature_stats': self.feature_stats
            }
            dump(model_data, filepath)
            print(f"Model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @staticmethod
    def load_model(filepath):
        """Load saved models and preprocessing components."""
        try:
            model_data = load(filepath)
            classifier = UnifiedHeartSoundClassifier()
            classifier.xgb_model = model_data['xgb_model']
            classifier.nn_model = model_data['nn_model']
            classifier.scaler = model_data['scaler']
            classifier.ensemble_model = model_data['ensemble_model']
            classifier.feature_stats = model_data['feature_stats']
            print(f"Model successfully loaded from {filepath}")
            return classifier
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def evaluate_model_performance(self, X, y):
        """Evaluate model performance with detailed metrics."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Dictionary to store all results
            detailed_results = {}
            
            # 1. XGBoost Evaluation
            print("\nEvaluating XGBoost...")
            self.xgb_model.fit(X_train, y_train)
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_metrics = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'classification_report': classification_report(y_test, xgb_pred),
                'confusion_matrix': confusion_matrix(y_test, xgb_pred)
            }
            detailed_results['XGBoost'] = xgb_metrics
            
            # 2. Neural Network Evaluation
            print("\nEvaluating Neural Network...")
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)
            
            self.build_nn_model((X_train.shape[1],), len(np.unique(y)))
            history = self.nn_model.fit(
                X_train, y_train_cat,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            nn_pred = np.argmax(self.nn_model.predict(X_test), axis=1)
            nn_metrics = {
                'accuracy': accuracy_score(y_test, nn_pred),
                'classification_report': classification_report(y_test, nn_pred),
                'confusion_matrix': confusion_matrix(y_test, nn_pred),
                'training_history': history.history
            }
            detailed_results['Neural Network'] = nn_metrics
            
            # 3. Ensemble Model Evaluation
            print("\nEvaluating Ensemble Model...")
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('xgb', self.xgb_model),
                    ('nn', LogisticRegression(multi_class='ovr', max_iter=1000))
                ],
                voting='soft'
            )
            self.ensemble_model.fit(X_train, y_train)
            ensemble_pred = self.ensemble_model.predict(X_test)
            ensemble_metrics = {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'classification_report': classification_report(y_test, ensemble_pred),
                'confusion_matrix': confusion_matrix(y_test, ensemble_pred)
            }
            detailed_results['Ensemble'] = ensemble_metrics
            
            return detailed_results
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return None

    def analyze_feature_importance(self, X, y):
        """Analyze and visualize feature importance."""
        try:
            # Train XGBoost for feature importance
            self.xgb_model.fit(X, y)
            
            # Get feature importance scores
            importance_scores = self.xgb_model.feature_importances_
            
            # Create feature names based on feature_stats
            feature_names = []
            for feature_type, dim in self.feature_stats.items():
                feature_names.extend([f"{feature_type}_{i}" for i in range(dim)])
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importance_scores)],
                'Importance': importance_scores
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.show()
            
            return importance_df
            
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
            return None

def main():
    """Main execution function."""
    try:
        # Initialize classifier
        classifier = UnifiedHeartSoundClassifier()
        
        # Set up data directory and labels
        data_dir = r"C:\Users\Ahmed Abd El Rahman\Desktop\programming\GP\model\(ORIGINAL)DATASET-HEARTSOUNDS\(ORIGINAL)DATASET-HEARTSOUNDS"  # Replace with your dataset path
        labels = ['abnormal', 'artifact', 'extrahls', 'extrastole', 'murmur', 'normal', 'special cases']
        
        print("Starting heart sound classification process...")
        
        # Prepare data
        print("\nPreparing and extracting features from data...")
        X, y = classifier.prepare_data(data_dir, labels)
        
        # Evaluate feature combinations
        print("\nEvaluating different feature combinations...")
        feature_results = classifier.evaluate_feature_combinations(X, y)
        
        # Print feature combination results
        print("\nFeature Combination Results:")
        for combo, accuracy in feature_results.items():
            print(f"{combo}: {accuracy:.4f}")
        
        # Perform detailed model evaluation
        print("\nPerforming detailed model evaluation...")
        detailed_results = classifier.evaluate_model_performance(X, y)
        
        # Print detailed results
        print("\n=== Detailed Model Performance ===")
        for model_name, metrics in detailed_results.items():
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("\nClassification Report:")
            print(metrics['classification_report'])
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
        # Analyze feature importance
        print("\nAnalyzing feature importance...")
        importance_df = classifier.analyze_feature_importance(X, y)
        
        # Save the model
        print("\nSaving model...")
        classifier.save_model('heart_sound_classifier.joblib')
        
        return classifier, detailed_results, importance_df
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None, None

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure warnings
    warnings.filterwarnings('ignore')
    
    # Run main function
    classifier, detailed_results, importance_df = main()
    
    if detailed_results is not None:
        print("\n=== Final Performance Summary ===")
        print("\nModel Accuracies:")
        for model_name, metrics in detailed_results.items():
            print(f"{model_name}: {metrics['accuracy']:.4f}")
        
        # Plot confusion matrices
        for model_name, metrics in detailed_results.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d')
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()