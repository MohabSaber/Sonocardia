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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import gc
import warnings
warnings.filterwarnings('ignore')

class UnifiedHeartSoundClassifier:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            tree_method='hist',  # More efficient for larger datasets
            enable_categorical=False,  # Explicitly disable categorical features
            eval_metric='mlogloss'
        )
        self.nn_model = None
        self.scaler = StandardScaler()
        self.ensemble_model = None
        
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
            from scipy.signal import butter, filtfilt
            nyquist = 0.5 * sr
            low = low_cut / nyquist
            high = high_cut / nyquist
            if high > 1.0:
                print("Warning: High cutoff frequency exceeds Nyquist frequency")
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
        try:
            if signal is None or len(signal) == 0:
                print("Error: Invalid signal for feature extraction")
                return None

            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            if np.any(np.isnan(mfccs)):
                print("Warning: NaN values in MFCC features")
                return None

            zcr = librosa.feature.zero_crossing_rate(y=signal)
            if np.any(np.isnan(zcr)):
                print("Warning: NaN values in ZCR features")
                return None

            rms = librosa.feature.rms(y=signal)
            if np.any(np.isnan(rms)):
                print("Warning: NaN values in RMS features")
                return None

            # Add spectral features for better classification
            spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

            features = {
                'mfccs': mfccs.mean(axis=1),
                'zcr': zcr.mean(),
                'rms_energy': rms.mean(),
                'spectral_centroid': spectral_centroid.mean(),
                'spectral_rolloff': spectral_rolloff.mean(),
                'spectral_bandwidth': spectral_bandwidth.mean()
            }

            feature_vector = np.concatenate([
                features['mfccs'],
                [features['zcr']],
                [features['rms_energy']],
                [features['spectral_centroid']],
                [features['spectral_rolloff']],
                [features['spectral_bandwidth']]
            ])

            if np.any(np.isnan(feature_vector)):
                print("Warning: NaN values in final feature vector")
                return None

            return feature_vector

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def prepare_data(self, data_dir, labels):
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        all_files = []
        all_labels = []
        processed_features = []
        valid_labels = []

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
            all_labels.extend([label] * len(files))

        if not all_files:
            raise ValueError("No valid audio files found in the data directory")

        data_df = pd.DataFrame(all_files, columns=['file_path', 'label'])
        data_df['label'] = pd.Categorical(data_df['label']).codes

        total_files = len(data_df)
        print(f"Found {total_files} files to process")

        for idx, (_, row) in enumerate(data_df.iterrows(), 1):
            try:
                print(f"\nProcessing file {idx}/{total_files}: {row['file_path']}")
                
                signal = self.preprocess_audio(row['file_path'])
                if signal is None:
                    print(f"Skipping file due to preprocessing failure")
                    continue

                features = self.extract_features(signal)
                if features is None:
                    print(f"Skipping file due to feature extraction failure")
                    continue

                processed_features.append(features)
                valid_labels.append(row['label'])

                if idx % 10 == 0:
                    print(f"Successfully processed {idx}/{total_files} files ({(idx/total_files)*100:.1f}%)")
                    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
                
                if idx % 50 == 0:
                    gc.collect()

            except Exception as e:
                print(f"Error processing file {row['file_path']}: {str(e)}")
                continue

        if not processed_features:
            raise ValueError("No features were successfully extracted from any files")

        X = np.array(processed_features)
        y = np.array(valid_labels)

        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        if np.any(np.isnan(X)):
            raise ValueError("Dataset contains NaN values")
        if len(X) != len(y):
            raise ValueError("Feature and label arrays have different lengths")

        X = self.scaler.fit_transform(X)
        return X, y

    def build_nn_model(self, input_shape, num_classes):
        self.nn_model = Sequential([
            Input(shape=input_shape),
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

    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = {}

        # Train XGBoost
        print("\nTraining XGBoost...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Create evaluation set
        eval_set = [(X_test, y_test)]
        
        try:
            self.xgb_model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=True
            )
            
            # Get best iteration
            best_iteration = self.xgb_model.best_iteration
            print(f"Best iteration: {best_iteration}")
            
            # Make predictions
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            results['XGBoost'] = xgb_accuracy
            print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

        except Exception as e:
            print(f"Error in XGBoost training: {str(e)}")
            print("Falling back to basic training without early stopping...")
            try:
                # Fallback training without early stopping
                self.xgb_model.fit(X_train, y_train)
                xgb_pred = self.xgb_model.predict(X_test)
                xgb_accuracy = accuracy_score(y_test, xgb_pred)
                results['XGBoost'] = xgb_accuracy
                print(f"XGBoost Accuracy (fallback): {xgb_accuracy:.4f}")
            except Exception as e:
                print(f"Error in fallback XGBoost training: {str(e)}")
                return None

        # Train Neural Network
        print("\nTraining Neural Network...")
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
        self.nn_model.fit(
            X_train, y_train_cat,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        nn_pred_prob = self.nn_model.predict(X_test)
        nn_pred = np.argmax(nn_pred_prob, axis=1)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        results['Neural Network'] = nn_accuracy
        print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

        # Build and evaluate ensemble model
        print("\nBuilding Ensemble Model...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('nn', LogisticRegression(solver='liblinear', multi_class='ovr'))
            ],
            voting='soft'
        )
        self.ensemble_model.fit(X_train, y_train)
        ensemble_pred = self.ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        results['Ensemble'] = ensemble_accuracy
        print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

        # Visualize results
        plt.figure(figsize=(8, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Model Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.show()

        # Feature importance plot for XGBoost
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.xgb_model, max_num_features=20)
        plt.title('XGBoost Feature Importance')
        plt.show()

        return results

    def save_model(self, filepath):
        model_data = {
            'xgb_model': self.xgb_model,
            'nn_model': self.nn_model,
            'scaler': self.scaler,
            'ensemble_model': self.ensemble_model
        }
        dump(model_data, filepath)

    @staticmethod
    def load_model(filepath):
        return load(filepath)

if __name__ == "__main__":
    classifier = UnifiedHeartSoundClassifier()
    
    data_dir = r"C:\Users\Ahmed Abd El Rahman\Desktop\GP\model\(ORIGINAL)DATASET-HEARTSOUNDS"
    labels = ['abnormal', 'artifact', 'extrahls', 'extrastole', 'murmur', 'normal', 'special cases']
    
    print("Preparing data...")
    X, y = classifier.prepare_data(data_dir, labels)
        
    print("Building neural network...")
    classifier.build_nn_model((X.shape[1],), len(np.unique(y)))
        
    print("Training and evaluating models...")
    results = classifier.train_and_evaluate(X, y)
        
    print("Saving model...")
    classifier.save_model('heart_sound_classifier.joblib')
        
    print("\nFinal Results:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")