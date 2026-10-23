"""
Speech Emotion Recognition (SER) - Model Training and Feature Extraction Pipeline
Task 3 - CodeAlpha Machine Learning Internship

This module provides:
1. Feature extraction from audio: MFCCs, Chroma, and Mel Spectrogram.
2. Realistic acoustic waveform synthesis for 6 target emotions:
   Calm, Happy, Sad, Angry, Fearful, Neutral.
3. Fallback synthetic feature generator for robust offline training.
4. Real dataset loader (supports RAVDESS and emotion-subfolder structures).
5. Model training pipeline with StandardScaler and MLPClassifier.
6. Export of trained model artifact to `emotion_model.pkl`.
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Target emotion classes
TARGET_EMOTIONS = ["Angry", "Calm", "Fearful", "Happy", "Neutral", "Sad"]

# RAVDESS emotion code mapping
RAVDESS_EMOTIONS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
}


def extract_features(audio_input, sr=22050, n_mfcc=40, n_mels=128):
    """
    Extract MFCCs, Chroma, and Mel-Spectrogram features from audio input.

    Parameters:
    -----------
    audio_input : str, file-like object, or tuple (np.ndarray, int)
        Path to audio file, file buffer, or (waveform, sample_rate) tuple.
    sr : int
        Target sample rate for resampling.
    n_mfcc : int
        Number of MFCC coefficients to compute (default: 40).
    n_mels : int
        Number of Mel bands to generate (default: 128).

    Returns:
    --------
    features : np.ndarray of shape (180,)
        Concatenated 1D feature vector:
        - 40 MFCC mean coefficients
        - 12 Chroma STFT mean values
        - 128 Mel Spectrogram mean energy bands
    """
    try:
        # Load audio if path or buffer is given
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            y, current_sr = audio_input
            if current_sr != sr:
                y = librosa.resample(y=y, orig_sr=current_sr, target_sr=sr)
        else:
            y, current_sr = librosa.load(audio_input, sr=sr, mono=True)

        # Handle silent or empty audio
        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return np.zeros(n_mfcc + 12 + n_mels, dtype=np.float32)

        # Normalize audio amplitude
        y = librosa.util.normalize(y)

        # Trim leading and trailing silence (top_db=25)
        trimmed_y, _ = librosa.effects.trim(y, top_db=25)
        if len(trimmed_y) > int(sr * 0.1):  # ensure at least 100ms
            y = trimmed_y

        # 1. MFCC (40 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        # 2. Chroma STFT (12 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 3. Mel-Spectrogram (128 features)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_mean = np.mean(mel, axis=1)

        # Concatenate into 180-dimensional feature vector
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean]).astype(np.float32)
        return features

    except Exception as e:
        # Safe fallback: return zero vector if extraction encounters an issue
        print(f"[Warning] Feature extraction failed: {e}")
        return np.zeros(n_mfcc + 12 + n_mels, dtype=np.float32)


def synthesize_emotional_waveform(emotion, duration=2.5, sr=22050):
    """
    Acoustic synthesis of human vocalization characteristics for 6 emotions.
    Simulates speech formants, fundamental frequency (F0), vibrato, jitter,
    dynamic loudness contours, and breath/noise.
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    rng = np.random.default_rng()

    # Emotion-specific acoustic parameters:
    # f0_base: pitch fundamental frequency
    # vibrato_rate, vibrato_depth: pitch instability / emotion tremolo
    # energy_mult: overall loudness
    # formant_boost: resonant vowel formant simulation
    # noise_level: aspiration / breathiness / aggression distortion
    if emotion == "Angry":
        f0 = 260.0 + rng.uniform(-20, 30)
        vibrato = 6.0 * np.sin(2 * np.pi * 7.5 * t)
        energy_env = np.clip(1.2 - 0.4 * t / duration, 0.5, 1.2)
        # Harsh harmonics (odd and even overtones)
        wave = (
            0.6 * np.sin(2 * np.pi * (f0 + vibrato) * t)
            + 0.5 * np.sin(2 * np.pi * 2 * (f0 + vibrato) * t)
            + 0.4 * np.sin(2 * np.pi * 3 * (f0 + vibrato) * t)
            + 0.3 * np.sin(2 * np.pi * 4 * (f0 + vibrato) * t)
        )
        # Burst noise
        noise = rng.normal(0, 0.08, num_samples)
        wave = (wave * energy_env) + noise

    elif emotion == "Happy":
        f0 = 230.0 + rng.uniform(-15, 25)
        # Rising intonation with joyful melodic swing
        f0_contour = f0 + 50.0 * np.sin(np.pi * t / duration)
        vibrato = 4.0 * np.sin(2 * np.pi * 5.5 * t)
        wave = (
            0.6 * np.sin(2 * np.pi * (f0_contour + vibrato) * t)
            + 0.4 * np.sin(2 * np.pi * 2 * (f0_contour + vibrato) * t)
            + 0.25 * np.sin(2 * np.pi * 3 * (f0_contour + vibrato) * t)
        )
        # Smooth lively envelope
        envelope = np.sin(np.pi * t / duration) ** 0.8
        wave = wave * envelope + rng.normal(0, 0.02, num_samples)

    elif emotion == "Sad":
        f0 = 135.0 + rng.uniform(-10, 10)
        # Downward pitch slope, subdued energy, slower decay
        f0_contour = f0 - 25.0 * (t / duration)
        vibrato = 1.5 * np.sin(2 * np.pi * 3.0 * t)
        wave = (
            0.7 * np.sin(2 * np.pi * (f0_contour + vibrato) * t)
            + 0.2 * np.sin(2 * np.pi * 2 * (f0_contour + vibrato) * t)
        )
        envelope = np.maximum(0.1, 0.9 - 0.7 * (t / duration))
        wave = wave * envelope + rng.normal(0, 0.01, num_samples)

    elif emotion == "Fearful":
        f0 = 275.0 + rng.uniform(-20, 20)
        # Rapid trembling vibrato (jitter/tremolo) and erratic energy
        vibrato = 12.0 * np.sin(2 * np.pi * 9.0 * t) + rng.normal(0, 3.0, num_samples)
        wave = (
            0.5 * np.sin(2 * np.pi * (f0 + vibrato) * t)
            + 0.3 * np.sin(2 * np.pi * 2 * (f0 + vibrato) * t)
            + 0.25 * np.sin(2 * np.pi * 3 * (f0 + vibrato) * t)
        )
        # Shivering envelope
        shiver = 0.7 + 0.3 * np.sin(2 * np.pi * 8.0 * t)
        noise = rng.normal(0, 0.06, num_samples)
        wave = (wave * shiver) + noise

    elif emotion == "Calm":
        f0 = 160.0 + rng.uniform(-10, 10)
        # Smooth, stable pitch with gentle bell-shaped envelope
        vibrato = 1.0 * np.sin(2 * np.pi * 3.5 * t)
        wave = (
            0.75 * np.sin(2 * np.pi * (f0 + vibrato) * t)
            + 0.25 * np.sin(2 * np.pi * 2 * (f0 + vibrato) * t)
            + 0.1 * np.sin(2 * np.pi * 3 * (f0 + vibrato) * t)
        )
        envelope = np.sin(np.pi * t / duration)
        wave = wave * envelope + rng.normal(0, 0.005, num_samples)

    else:  # Neutral
        f0 = 180.0 + rng.uniform(-10, 10)
        # Steady, conversational cadence
        vibrato = 2.0 * np.sin(2 * np.pi * 4.0 * t)
        wave = (
            0.7 * np.sin(2 * np.pi * (f0 + vibrato) * t)
            + 0.3 * np.sin(2 * np.pi * 2 * (f0 + vibrato) * t)
            + 0.15 * np.sin(2 * np.pi * 3 * (f0 + vibrato) * t)
        )
        envelope = np.clip(1.0 - np.abs(2 * t / duration - 1.0) ** 2, 0.1, 1.0)
        wave = wave * envelope + rng.normal(0, 0.015, num_samples)

    # Normalize audio
    wave = wave / (np.max(np.abs(wave)) + 1e-6)
    return wave.astype(np.float32)


def generate_synthetic_dataset(samples_per_emotion=250, sr=22050):
    """
    Generate synthetic dataset of audio features for training.
    Creates acoustic waveforms using emotional synthesis, then passes each
    waveform through `extract_features` to ensure identical feature extraction.
    """
    print(f"\n[Info] Generating synthetic dataset ({samples_per_emotion} samples/emotion across {len(TARGET_EMOTIONS)} emotions)...")
    X = []
    y = []

    start_time = time.time()
    for emotion_idx, emotion in enumerate(TARGET_EMOTIONS):
        print(f"  Generating synthetic speech profiles for emotion: {emotion} ({emotion_idx+1}/{len(TARGET_EMOTIONS)})")
        for i in range(samples_per_emotion):
            # Vary duration slightly (1.8s to 3.2s)
            duration = np.random.uniform(1.8, 3.2)
            waveform = synthesize_emotional_waveform(emotion, duration=duration, sr=sr)
            feat = extract_features((waveform, sr), sr=sr)
            X.append(feat)
            y.append(emotion)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    elapsed = time.time() - start_time
    print(f"[Success] Generated {len(X)} synthetic samples with {X.shape[1]} features in {elapsed:.1f}s.")
    return X, y


def generate_demo_samples(output_dir="demo_samples", sr=22050):
    """
    Generate realistic demo .wav files for each emotion so the web UI
    can instantly play and analyze them without external downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Info] Generating demo audio samples in '{output_dir}/'...")
    demo_files = {}
    for emotion in TARGET_EMOTIONS:
        filename = f"{emotion.lower()}_sample.wav"
        filepath = os.path.join(output_dir, filename)
        waveform = synthesize_emotional_waveform(emotion, duration=3.0, sr=sr)
        sf.write(filepath, waveform, sr)
        demo_files[emotion] = filepath
        print(f"  Saved demo sample: {filepath}")
    return demo_files


def load_real_dataset(data_dir, sr=22050):
    """
    Load real speech emotion audio dataset from a directory.
    Supports:
    1. Emotion subdirectories: `data_dir/angry/*.wav`, `data_dir/happy/*.wav`, etc.
    2. RAVDESS filename format: `03-01-05-01-01-01-01.wav` (3rd token is emotion code)
    """
    print(f"\n[Info] Scanning directory for real audio dataset: '{data_dir}'...")
    X, y = [], []

    if not os.path.exists(data_dir):
        print(f"[Warning] Directory '{data_dir}' not found.")
        return None, None

    audio_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        print("[Warning] No compatible audio files found in directory.")
        return None, None

    print(f"[Info] Found {len(audio_files)} audio files. Extracting features...")
    for idx, fpath in enumerate(audio_files):
        fname = os.path.basename(fpath)
        emotion = None

        # Check for RAVDESS naming convention
        parts = fname.split("-")
        if len(parts) >= 7 and parts[0] == "03" and parts[1] == "01":
            code = parts[2]
            emotion = RAVDESS_EMOTIONS.get(code)

        # Check parent folder name
        if not emotion:
            parent = os.path.basename(os.path.dirname(fpath)).capitalize()
            for target in TARGET_EMOTIONS:
                if target.lower() in parent.lower():
                    emotion = target
                    break

        # Check filename directly
        if not emotion:
            for target in TARGET_EMOTIONS:
                if target.lower() in fname.lower():
                    emotion = target
                    break

        if emotion and emotion in TARGET_EMOTIONS:
            feat = extract_features(fpath, sr=sr)
            X.append(feat)
            y.append(emotion)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(audio_files):
            print(f"  Processed {idx + 1}/{len(audio_files)} files...")

    if len(X) == 0:
        print("[Warning] Could not map audio files to target emotions.")
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"[Success] Extracted features for {len(X)} real audio samples.")
    return X, y


def train_model(X, y, model_path="emotion_model.pkl"):
    """
    Train a Multi-Layer Perceptron (MLP) Classifier with StandardScaler,
    evaluate on held-out test split, and save model artifact.
    """
    print(f"\n[Info] Splitting dataset into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Standardize features
    print("[Info] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP Neural Network Classifier
    print("[Info] Training Multi-Layer Perceptron (MLPClassifier)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=350,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )

    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"[Success] Model training finished in {train_time:.2f}s.")

    # Evaluate
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f" Model Test Accuracy: {acc * 100:.2f}%")
    print(f"==========================================")
    print("\nClassification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Package artifact
    artifact = {
        "model": mlp,
        "scaler": scaler,
        "classes": sorted(list(set(y))),
        "feature_dim": X.shape[1],
        "accuracy": float(acc),
        "report": report_dict,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_count": len(X),
    }

    joblib.dump(artifact, model_path)
    print(f"[Success] Saved trained model artifact to: '{model_path}'")
    return artifact


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Model Trainer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to real audio dataset directory (default: 'data')",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic audio generation",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=250,
        help="Number of synthetic samples per emotion (default: 250)",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="emotion_model.pkl",
        help="Output model path (default: 'emotion_model.pkl')",
    )
    args = parser.parse_args()

    # Step 1: Generate Demo Samples for immediate UI testing
    generate_demo_samples(output_dir="demo_samples")

    # Step 2: Acquire Data (Real or Fallback Synthetic)
    X, y = None, None
    if not args.synthetic and os.path.exists(args.data_dir):
        X, y = load_real_dataset(args.data_dir)

    if X is None or len(X) == 0:
        print("[Info] No real dataset provided or found. Utilizing fallback synthetic audio generator.")
        X, y = generate_synthetic_dataset(samples_per_emotion=args.samples)

    # Step 3: Train & Save Model
    train_model(X, y, model_path=args.model_out)
    print("\n[Done] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
