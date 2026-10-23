"""
EmotiVoice AI - Speech Emotion Recognition (SER) Web Application
Task 3 - CodeAlpha Machine Learning Internship

Interactive Streamlit Web Dashboard:
- Audio file upload (.wav, .mp3, .ogg, .flac)
- Real-time microphone recording (st.audio_input)
- Instant Demo sample test suite
- Audio waveform & Mel-Spectrogram visualization
- Probability breakdown across 6 target emotions
- Acoustic feature inspector (MFCCs, Chroma, Mel bands)
"""

import io
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import joblib
import streamlit as st

# Import training/feature extraction utilities
from train import extract_features, synthesize_emotional_waveform, train_model, generate_synthetic_dataset

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="EmotiVoice AI - Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Emotion Design System: Palette, Emojis, and Descriptions
EMOTION_CONFIG = {
    "Angry": {
        "emoji": "😠",
        "color": "#FF4D4D",
        "gradient": "linear-gradient(135deg, #FF4D4D, #B71C1C)",
        "tone": "High vocal tension, rapid attack, loud acoustic energy, and aggressive harmonics.",
    },
    "Calm": {
        "emoji": "😌",
        "color": "#00BFA5",
        "gradient": "linear-gradient(135deg, #00BFA5, #00695C)",
        "tone": "Stable fundamental frequency, gentle attack envelope, relaxed acoustic cadence.",
    },
    "Fearful": {
        "emoji": "😨",
        "color": "#9C27B0",
        "gradient": "linear-gradient(135deg, #9C27B0, #4A148C)",
        "tone": "Elevated pitch jitter, breathy spectral tilt, rapid trembling vocal vibrato.",
    },
    "Happy": {
        "emoji": "😄",
        "color": "#FFB300",
        "gradient": "linear-gradient(135deg, #FFB300, #E65100)",
        "tone": "Lively pitch modulation, bright spectral centroid, dynamic joyful intonation.",
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#78909C",
        "gradient": "linear-gradient(135deg, #78909C, #37474F)",
        "tone": "Balanced conversational speaking rate, moderate energy, flat chroma variance.",
    },
    "Sad": {
        "emoji": "😢",
        "color": "#29B6F6",
        "gradient": "linear-gradient(135deg, #29B6F6, #0D47A1)",
        "tone": "Subdued fundamental pitch, downward inflection, muffled high-frequency energy.",
    },
}

# Inject Custom CSS for Premium Glassmorphic UI
st.markdown(
    """
    <style>
    /* Global styling */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    .hero-card {
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #FFFFFF;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .hero-emoji {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 8px;
    }
    .hero-emotion {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 9999px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    .tone-desc {
        font-size: 0.92rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        line-height: 1.45;
    }
    .stat-badge {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model Loading & Caching
# ==============================================================================
@st.cache_resource
def load_emotion_model(model_path="emotion_model.pkl"):
    """
    Load trained model artifact containing classifier, scaler, classes,
    and evaluation metadata.
    """
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None


# ==============================================================================
# Audio Processing Utilities
# ==============================================================================
def load_and_preprocess_audio(audio_source, sr=22050, trim_silence=True, top_db=25):
    """
    Load audio from file/buffer, convert to mono, resample, and optionally trim silence.
    Returns: (waveform, sample_rate, original_duration, processed_duration)
    """
    try:
        if isinstance(audio_source, (str, io.BytesIO)):
            y, current_sr = librosa.load(audio_source, sr=sr, mono=True)
        elif hasattr(audio_source, "read"):
            # Handle Streamlit UploadedFile or AudioBuffer
            audio_bytes = audio_source.read()
            buf = io.BytesIO(audio_bytes)
            y, current_sr = librosa.load(buf, sr=sr, mono=True)
        else:
            return None, None, 0, 0

        orig_duration = len(y) / sr

        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None, None, 0, 0

        # Normalize
        y = librosa.util.normalize(y)

        # Silence trimming
        if trim_silence:
            trimmed_y, _ = librosa.effects.trim(y, top_db=top_db)
            if len(trimmed_y) > int(sr * 0.15):  # keep if >= 150ms
                y = trimmed_y

        proc_duration = len(y) / sr
        return y, sr, orig_duration, proc_duration

    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None, 0, 0


def compute_audio_stats(y, sr):
    """Compute physical acoustic statistics for feature insights."""
    duration = len(y) / sr
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    return {
        "duration": duration,
        "rms_energy": float(rms),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid": float(spec_cent),
        "spectral_rolloff": float(spec_rolloff),
    }


def plot_waveform_and_spectrogram(y, sr):
    """
    Generate professional dark-themed dual acoustic plots:
    1. Time-Domain Waveform with RMS envelope
    2. Mel-Spectrogram (Frequency-Domain)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), dpi=120)
    fig.patch.set_facecolor("#0F172A")

    # 1. Waveform Plot
    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax1.set_facecolor("#1E293B")
    ax1.plot(time_axis, y, color="#38BDF8", linewidth=1.0, alpha=0.9, label="Amplitude")
    ax1.fill_between(time_axis, y, -y, color="#38BDF8", alpha=0.15)
    ax1.set_title("Audio Signal Waveform (Time Domain)", color="#F1F5F9", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Amplitude", color="#94A3B8", fontsize=9)
    ax1.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=9)
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
    ax1.set_xlim(0, time_axis[-1])
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # 2. Mel-Spectrogram Plot
    ax2.set_facecolor("#1E293B")
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(
        mel_spec_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        fmax=8000,
        ax=ax2,
        cmap="magma",
    )
    ax2.set_title("Mel-Spectrogram (Acoustic Frequency vs Time)", color="#F1F5F9", fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Hz", color="#94A3B8", fontsize=9)
    ax2.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=9)
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#94A3B8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94A3B8", size=8)

    plt.tight_layout()
    return fig


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">🎙️ EmotiVoice AI — Speech Emotion Recognition</h1>
            <p class="sub-title">
                Production-grade vocal acoustic analysis powered by <strong>Librosa</strong> feature extraction
                (MFCCs, Chroma, Mel-Spectrogram) & Deep <strong>MLP Neural Network</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Model Artifact
    artifact = load_emotion_model("emotion_model.pkl")

    # --------------------------------------------------------------------------
    # Sidebar: Model Health & Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("⚙️ System Status & Controls")
        
        if artifact is not None:
            st.success("🟢 Model Loaded: Ready")
            st.markdown(
                f"""
                - **Accuracy**: `{artifact.get('accuracy', 1.0) * 100:.1f}%`
                - **Features**: `{artifact.get('feature_dim', 180)}` Acoustic Dims
                - **Classes**: `{len(artifact.get('classes', []))}` Emotions
                - **Trained on**: `{artifact.get('samples_count', 900)}` samples
                """
            )
        else:
            st.warning("⚠️ Model artifact not found (`emotion_model.pkl`).")
            if st.button("🚀 Train Model Now", use_container_width=True):
                with st.spinner("Synthesizing acoustic audio and training MLP neural network..."):
                    X, y = generate_synthetic_dataset(samples_per_emotion=120)
                    train_model(X, y, "emotion_model.pkl")
                    st.success("Model trained successfully! Reloading...")
                    st.cache_resource.clear()
                    st.rerun()

        st.divider()
        st.subheader("🎚️ Audio Preprocessing")
        trim_silence = st.checkbox("Silence Trimming", value=True, help="Trims leading and trailing silence.")
        top_db = st.slider("Silence Threshold (dB)", min_value=10, max_value=40, value=25, step=5,
                           help="Lower value trims quieter sounds; 25 dB is recommended.")
        target_sr = st.selectbox("Resampling Rate (Hz)", [22050, 16000, 44100], index=0)

        st.divider()
        with st.expander("📚 Emotion Acoustic Theory"):
            st.markdown(
                """
                - **Angry**: High pitch, strong high harmonics, fast speech attack.
                - **Happy**: Dynamic pitch variation, bright spectral centroid.
                - **Sad**: Flat low pitch, downward inflection, muffled energy.
                - **Fearful**: Fast jitter/vibrato, breathy spectral tilt, tremors.
                - **Calm**: Low pitch variation, smooth harmonic envelope.
                - **Neutral**: Moderate cadence, flat chroma, standard speech tempo.
                """
            )

        with st.expander("🔄 Retrain Model"):
            samples_slider = st.slider("Synthetic Samples / Emotion", 50, 200, 100, 25)
            if st.button("Retrain with Selected Samples", use_container_width=True):
                with st.spinner("Retraining neural network model..."):
                    X, y = generate_synthetic_dataset(samples_per_emotion=samples_slider)
                    train_model(X, y, "emotion_model.pkl")
                    st.cache_resource.clear()
                    st.success("Retrained successfully!")
                    st.rerun()

    # --------------------------------------------------------------------------
    # Audio Input Selection Tabs
    # --------------------------------------------------------------------------
    tab_upload, tab_record, tab_demo = st.tabs([
        "📁 Option A: Upload Audio File",
        "🎙️ Option B: Record via Microphone",
        "🎵 Option C: Instant Demo Showcase",
    ])

    audio_bytes_data = None
    audio_label = ""

    # Tab 1: Upload Audio File
    with tab_upload:
        st.markdown("Upload any pre-recorded speech file (`.wav`, `.mp3`, `.ogg`, `.flac`).")
        uploaded_file = st.file_uploader(
            "Choose a voice audio file",
            type=["wav", "mp3", "ogg", "flac"],
            help="High-quality vocal recordings provide best emotion accuracy.",
            key="file_uploader",
        )
        if uploaded_file is not None:
            audio_bytes_data = uploaded_file.getvalue()
            audio_label = f"Uploaded File: {uploaded_file.name}"

    # Tab 2: Microphone Recording (Streamlit 1.39 native audio input)
    with tab_record:
        st.markdown("Speak clearly into your microphone to capture real-time speech emotion.")
        recorded_audio = st.audio_input("Click the microphone to start recording", key="audio_recorder")
        if recorded_audio is not None:
            audio_bytes_data = recorded_audio.getvalue()
            audio_label = "Live Microphone Recording"

    # Tab 3: Built-in Demo Samples
    with tab_demo:
        st.markdown("Test instant acoustic voice profiles for all 6 target emotions with a single click:")
        demo_cols = st.columns(6)
        emotions_list = ["Angry", "Calm", "Fearful", "Happy", "Neutral", "Sad"]
        
        for idx, em in enumerate(emotions_list):
            with demo_cols[idx]:
                cfg = EMOTION_CONFIG[em]
                if st.button(f"{cfg['emoji']} {em}", use_container_width=True, key=f"demo_btn_{em}"):
                    demo_path = os.path.join("demo_samples", f"{em.lower()}_sample.wav")
                    if not os.path.exists(demo_path):
                        # Generate if missing
                        os.makedirs("demo_samples", exist_ok=True)
                        wave = synthesize_emotional_waveform(em, duration=3.0, sr=target_sr)
                        sf.write(demo_path, wave, target_sr)
                    
                    with open(demo_path, "rb") as f:
                        st.session_state["active_demo_bytes"] = f.read()
                        st.session_state["active_demo_label"] = f"Demo Preset: {em}"

        if "active_demo_bytes" in st.session_state and audio_bytes_data is None:
            audio_bytes_data = st.session_state["active_demo_bytes"]
            audio_label = st.session_state.get("active_demo_label", "Demo Sample")

    # --------------------------------------------------------------------------
    # Inference & Visualization Section
    # --------------------------------------------------------------------------
    if audio_bytes_data is not None:
        st.divider()
        st.markdown(f"### 🔍 Analysis for: `{audio_label}`")

        # Audio Player
        col_player, col_stats1, col_stats2, col_stats3 = st.columns([2, 1, 1, 1])
        with col_player:
            st.audio(audio_bytes_data)

        # Preprocess Audio
        with st.spinner("Extracting Librosa acoustic features (MFCC, Chroma, Mel-Spectrogram)..."):
            y, sr, orig_dur, proc_dur = load_and_preprocess_audio(
                io.BytesIO(audio_bytes_data),
                sr=target_sr,
                trim_silence=trim_silence,
                top_db=top_db,
            )

        if y is None or len(y) < int(target_sr * 0.15):
            st.warning("⚠️ The input audio is silent or too short (< 150ms). Please provide speech with audible vocalization.")
            return

        # Display Signal Statistics
        stats = compute_audio_stats(y, sr)
        with col_stats1:
            st.markdown(
                f"""<div class="stat-badge">
                    <div class="stat-val">{proc_dur:.2f}s</div>
                    <div class="stat-label">Duration</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_stats2:
            st.markdown(
                f"""<div class="stat-badge">
                    <div class="stat-val">{stats['rms_energy']:.3f}</div>
                    <div class="stat-label">RMS Energy</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_stats3:
            st.markdown(
                f"""<div class="stat-badge">
                    <div class="stat-val">{stats['zero_crossing_rate']:.3f}</div>
                    <div class="stat-label">Zero Crossing</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Extract features for classifier
        features = extract_features((y, sr), sr=sr)

        # Perform ML Model Inference
        if artifact is not None:
            model = artifact["model"]
            scaler = artifact["scaler"]
            classes = artifact["classes"]

            features_scaled = scaler.transform([features])
            probabilities = model.predict_proba(features_scaled)[0]

            # Primary Prediction
            top_idx = int(np.argmax(probabilities))
            predicted_emotion = classes[top_idx]
            top_confidence = probabilities[top_idx] * 100.0

            cfg = EMOTION_CONFIG.get(predicted_emotion, EMOTION_CONFIG["Neutral"])

            # Hero Prediction Card & Probability Chart
            st.markdown("<br>", unsafe_allow_html=True)
            col_hero, col_probs = st.columns([1, 1.25])

            with col_hero:
                st.markdown(
                    f"""
                    <div class="hero-card" style="background: {cfg['gradient']};">
                        <div class="hero-emoji">{cfg['emoji']}</div>
                        <div class="hero-emotion">{predicted_emotion}</div>
                        <div class="confidence-badge">🎯 {top_confidence:.1f}% Confidence</div>
                        <p class="tone-desc"><strong>Acoustic Signature:</strong><br>{cfg['tone']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_probs:
                st.markdown("#### 📊 Emotion Confidence Breakdown")
                prob_df = pd.DataFrame({
                    "Emotion": classes,
                    "Probability (%)": [p * 100.0 for p in probabilities],
                    "Color": [EMOTION_CONFIG.get(c, {}).get("color", "#38BDF8") for c in classes],
                }).sort_values(by="Probability (%)", ascending=False)

                # Custom stylized probability bars
                for _, row in prob_df.iterrows():
                    em = row["Emotion"]
                    p = row["Probability (%)"]
                    c = row["Color"]
                    emoji = EMOTION_CONFIG.get(em, {}).get("emoji", "")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-bottom: 3px;">
                                <span>{emoji} {em}</span>
                                <span style="color: {c};">{p:.1f}%</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 10px; overflow: hidden;">
                                <div style="background: {c}; width: {p}%; height: 100%; border-radius: 6px; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Acoustic Waveform and Spectrogram Visualizations
            st.markdown("---")
            st.markdown("### 📈 Acoustic Waveform & Spectral Visualizations")
            spec_fig = plot_waveform_and_spectrogram(y, sr)
            st.pyplot(spec_fig)

            # Feature Deep Dive Expander
            with st.expander("🔬 Deep Dive: Extracted Acoustic Feature Vector (180 Dims)"):
                st.markdown(
                    """
                    Feature representations extracted from speech signal:
                    - **MFCCs (1-40)**: Captures vocal tract shape and phoneme envelopes.
                    - **Chroma STFT (12)**: Captures harmonic energy distribution across musical pitch classes.
                    - **Mel Spectrogram (1-128)**: Perceptual frequency bands modeling human auditory perception.
                    """
                )
                mfcc_features = features[:40]
                chroma_features = features[40:52]
                mel_features = features[52:180]

                tab_mfcc, tab_chroma, tab_mel = st.tabs(["MFCC Heatmap", "Chroma Distribution", "Mel Energy Spectrum"])

                with tab_mfcc:
                    fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 2.5), dpi=100)
                    fig_mfcc.patch.set_facecolor("#0F172A")
                    ax_mfcc.set_facecolor("#1E293B")
                    ax_mfcc.bar(range(1, 41), mfcc_features, color="#38BDF8", alpha=0.85)
                    ax_mfcc.set_title("Mean MFCC Coefficients (1 to 40)", color="#F1F5F9", fontsize=10, fontweight="bold")
                    ax_mfcc.set_xlabel("Coefficient Index", color="#94A3B8", fontsize=8)
                    ax_mfcc.set_ylabel("Value", color="#94A3B8", fontsize=8)
                    ax_mfcc.tick_params(colors="#94A3B8", labelsize=8)
                    ax_mfcc.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
                    st.pyplot(fig_mfcc)

                with tab_chroma:
                    pitch_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                    fig_chroma, ax_chroma = plt.subplots(figsize=(10, 2.5), dpi=100)
                    fig_chroma.patch.set_facecolor("#0F172A")
                    ax_chroma.set_facecolor("#1E293B")
                    ax_chroma.bar(pitch_labels, chroma_features, color="#A855F7", alpha=0.85)
                    ax_chroma.set_title("Chroma STFT Pitch Energy Distribution (12 Semitones)", color="#F1F5F9", fontsize=10, fontweight="bold")
                    ax_chroma.set_xlabel("Pitch Class", color="#94A3B8", fontsize=8)
                    ax_chroma.set_ylabel("Mean Magnitude", color="#94A3B8", fontsize=8)
                    ax_chroma.tick_params(colors="#94A3B8", labelsize=8)
                    ax_chroma.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
                    st.pyplot(fig_chroma)

                with tab_mel:
                    fig_mel, ax_mel = plt.subplots(figsize=(10, 2.5), dpi=100)
                    fig_mel.patch.set_facecolor("#0F172A")
                    ax_mel.set_facecolor("#1E293B")
                    ax_mel.plot(mel_features, color="#F59E0B", linewidth=1.5)
                    ax_mel.fill_between(range(len(mel_features)), mel_features, color="#F59E0B", alpha=0.2)
                    ax_mel.set_title("Mel Spectrogram Mean Energy Bands (128 Mel Bins)", color="#F1F5F9", fontsize=10, fontweight="bold")
                    ax_mel.set_xlabel("Mel Band Index", color="#94A3B8", fontsize=8)
                    ax_mel.set_ylabel("Power", color="#94A3B8", fontsize=8)
                    ax_mel.tick_params(colors="#94A3B8", labelsize=8)
                    ax_mel.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
                    st.pyplot(fig_mel)

        else:
            st.error("Model artifact could not be loaded. Please train or provide `emotion_model.pkl`.")

    else:
        # Empty State / Landing Guide
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 16px; padding: 40px; text-align: center; margin-top: 20px;">
                <div style="font-size: 3rem; margin-bottom: 12px;">🎧</div>
                <h3 style="color: #F1F5F9; margin-bottom: 8px;">No Audio Sample Selected Yet</h3>
                <p style="color: #94A3B8; max-width: 550px; margin: 0 auto 20px auto; font-size: 0.95rem;">
                    Upload an audio file, record your voice via the microphone, or click any of the
                    <strong>Instant Demo Showcase</strong> buttons above to experience real-time emotion recognition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
