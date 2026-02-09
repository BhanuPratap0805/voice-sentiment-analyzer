import streamlit as st
import whisper
from transformers import pipeline
import os
import tempfile
import plotly.express as px

# Check if running in a cloud environment (e.g., Streamlit Cloud) where system level deps might be an issue.
# For Whisper, we need ffmpeg. On Streamlit Cloud, this is usually handled by `packages.txt` if needed, 
# but `whisper` might just work if we rely on python bindings.
# Let's trust standard Streamlit Cloud environment or `packages.txt`.

st.set_page_config(
    page_title="Voice Sentiment Analyzer",
    page_icon="🎙️",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Cached Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once and cache it."""
    print("[*] Loading Whisper speech-recognition model (base)...")
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    """Load Emotion model once and cache it."""
    print("[*] Loading emotion classifier...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

# Load models
with st.spinner("Loading models (this may take a moment on first run)..."):
    asr_model = load_whisper_model()
    emotion_clf = load_emotion_model()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

def _check_excitement(scores: dict[str, float]) -> tuple[bool, float]:
    """Detect excitement from high joy + surprise combination."""
    joy = scores.get("joy", 0)
    surprise = scores.get("surprise", 0)
    if (joy >= 0.25 and surprise >= 0.12) or (surprise >= 0.25 and joy >= 0.12):
        return True, round(joy * 0.6 + surprise * 0.4, 3)
    return False, 0.0

# ---------------------------------------------------------------------------
# Main App UI
# ---------------------------------------------------------------------------
st.title("🎙️ Voice Sentiment Analyzer")
st.markdown("""
Upload an audio file to transcribe speech and analyze emotions.
Supported formats: WAV, MP3, M4A, OGG
""")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Audio"):
        try:
            with st.spinner("Transcribing and Analyzing..."):
                # Save uploaded file to temp
                # We need a file path for whisper.load_audio or transcribe
                suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # --- Transcribe ---
                    result = asr_model.transcribe(tmp_path)
                    raw_segments = result.get("segments", [])
                    full_text = result.get("text", "")
                    
                    if not raw_segments:
                        st.error("No speech detected. Try a clearer recording.")
                        st.stop()

                    duration = raw_segments[-1]["end"]
                    
                    # --- Emotion Analysis ---
                    segments = []
                    emo_dur: dict[str, float] = {}

                    for seg in raw_segments:
                        text = seg["text"].strip()
                        if not text:
                            continue
                        
                        emotions = emotion_clf(text)[0]
                        # emotions is a list of dicts: [{'label': 'joy', 'score': 0.9}, ...]
                        score_dict = {e["label"]: e["score"] for e in emotions}
                        
                        dominant = max(emotions, key=lambda e: e["score"])
                        
                        # Check excitement
                        is_excited, exc_conf = _check_excitement(score_dict)
                        if is_excited:
                            emo_label = "excitement"
                            confidence = exc_conf
                        else:
                            emo_label = dominant["label"]
                            confidence = dominant["score"]

                        # Accumulate duration for distribution
                        seg_dur = seg["end"] - seg["start"]
                        emo_dur[emo_label] = emo_dur.get(emo_label, 0) + seg_dur

                        segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": text,
                            "emotion": emo_label,
                            "confidence": confidence,
                            "scores": score_dict # keep full scores if needed
                        })

                    # --- Display Results ---
                    
                    # 1. Overview Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("Duration", f"{duration:.2f}s")
                    c1.metric("Word Count", len(full_text.split()))
                    
                    # dominant overall emotion (by duration)
                    if emo_dur:
                        overall_dominant = max(emo_dur, key=emo_dur.get)
                        c2.metric("Dominant Emotion", overall_dominant.capitalize())

                    st.divider()

                    # 2. Emotion Distribution Chart
                    st.subheader("Emotion Distribution (by time)")
                    if emo_dur:
                        data = [{"Emotion": k, "Duration (s)": v} for k, v in emo_dur.items()]
                        fig = px.pie(data, values="Duration (s)", names="Emotion", 
                                     color="Emotion", hole=0.4,
                                     title="Emotion Share")
                        st.plotly_chart(fig, use_container_width=True)

                    # 3. Transcript with Emotions
                    st.subheader("Transcript & Sentiment Stream")
                    
                    # Create a more visual timeline or just a list
                    for s in segments:
                        start_fmt = _fmt(s['start'])
                        end_fmt = _fmt(s['end'])
                        emoji = {
                            "joy": "😄", "sadness": "😢", "anger": "😡", 
                            "fear": "​​​​😨", "neutral": "😐", "surprise": "😲", 
                            "disgust": "🤢", "excitement": "🤩"
                        }.get(s['emotion'], "😶")
                        
                        # Color coding based on emotion could be done with st.markdown and HTML/CSS but let's keep it simple
                        with st.expander(f"{start_fmt} - {end_fmt} | {emoji} {s['emotion'].capitalize()} ({s['confidence']:.0%})"):
                            st.write(f"**\"{s['text']}\"**")
                            # Show detailed scores for this segment in a chart?
                            # scores_df = [{"Emotion": k, "Score": v} for k, v in s['scores'].items()]
                            # st.bar_chart(scores_df, x="Emotion", y="Score")

                    # 4. JSON Export
                    st.subheader("Raw Data")
                    with st.expander("View Full JSON Output"):
                        st.json({
                            "duration": duration,
                            "segments": segments,
                            "distribution": {k: round(v, 2) for k, v in emo_dur.items()}
                        })

                finally:
                    # Cleanup temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

