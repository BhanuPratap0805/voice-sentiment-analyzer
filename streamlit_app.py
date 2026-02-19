import streamlit as st
import streamlit.components.v1 as components
import whisper
from transformers import pipeline
import os
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Voice Sentiment Analyzer",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Luxury Minimalistic CSS — Black/Charcoal + Gold/Silver accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep: #08080c;
    --bg-card: rgba(14, 14, 20, 0.7);
    --glass-border: rgba(255, 255, 255, 0.04);
    --glass-bg: rgba(16, 16, 24, 0.55);
    --text-primary: #f0f0f4;
    --text-secondary: #6b6b80;
    --text-muted: #45455a;
    --accent-gold: #c9a96e;
    --accent-gold-dim: rgba(201, 169, 110, 0.15);
    --accent-silver: #a8a8b8;
    --accent-warm: #e8c885;
    --border-subtle: rgba(201, 169, 110, 0.08);
}

.stApp {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-deep) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== Interactive Canvas Container ===== */
#music-canvas-host {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}

/* ===== Hero Header ===== */
.hero-header {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    border-radius: 0;
    border-bottom: 1px solid var(--border-subtle);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    background: transparent;
    border: 1px solid rgba(201, 169, 110, 0.2);
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--accent-gold);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 300;
    color: var(--text-primary);
    margin-bottom: 0.8rem;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.hero-title span {
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-warm));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: var(--text-secondary);
    font-weight: 300;
    letter-spacing: 0.2px;
    max-width: 550px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ===== Thin Gold Line ===== */
.gold-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
    margin: 1.5rem auto;
}

/* ===== Glass Card ===== */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
    position: relative;
}

.glass-card:hover {
    border-color: rgba(201, 169, 110, 0.1);
    box-shadow: 0 16px 48px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}

.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ===== Metric Cards ===== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}

@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}

.metric-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
    position: relative;
}

.metric-card:hover {
    border-color: rgba(201, 169, 110, 0.12);
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.3);
}

.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.6rem;
    display: block;
    opacity: 0.7;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 200;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    letter-spacing: -1px;
}

.metric-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-top: 0.4rem;
}

/* ===== Dominant Emotion Hero ===== */
.emotion-hero {
    text-align: center;
    padding: 3rem 1rem;
    margin: 2rem 0;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}

.emotion-hero::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,169,110,0.04) 0%, transparent 70%);
}

.emotion-emoji-big {
    font-size: 4rem;
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 1;
    animation: gentleFloat 5s ease-in-out infinite;
}

@keyframes gentleFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.emotion-label-big {
    font-size: 1.4rem;
    font-weight: 300;
    color: var(--text-primary);
    position: relative;
    z-index: 1;
    text-transform: uppercase;
    letter-spacing: 4px;
}

.emotion-sublabel {
    font-size: 0.7rem;
    color: var(--text-muted);
    position: relative;
    z-index: 1;
    margin-top: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 500;
}

/* ===== Section Divider ===== */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 3rem 0;
    border: none;
}

/* ===== Emotion Pills ===== */
.emotion-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin: 1rem 0;
    justify-content: center;
}

.emotion-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 400;
    border: 1px solid var(--glass-border);
    background: rgba(255,255,255,0.01);
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    cursor: default;
}

.emotion-pill:hover {
    border-color: rgba(201, 169, 110, 0.15);
    background: rgba(201, 169, 110, 0.03);
    transform: translateY(-2px);
}

.pill-dur {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
}

/* ===== Change Points ===== */
.change-point {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.01);
    border-left: 1px solid var(--accent-gold);
    margin-bottom: 0.6rem;
    transition: all 0.4s ease;
}

.change-point:hover {
    background: rgba(201, 169, 110, 0.02);
    padding-left: 1.6rem;
}

.change-time {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--accent-gold);
    font-size: 0.85rem;
    min-width: 50px;
}

.change-arrow {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ===== Status Badge ===== */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: transparent;
    border: 1px solid rgba(201, 169, 110, 0.2);
    border-radius: 6px;
    color: var(--accent-gold);
    font-weight: 500;
    font-size: 0.8rem;
    margin-bottom: 1.5rem;
    letter-spacing: 1px;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--glass-border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 400;
    font-family: 'Inter', sans-serif;
    padding: 10px 24px;
    font-size: 0.85rem;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-gold-dim) !important;
    border: 1px solid rgba(201, 169, 110, 0.15) !important;
    color: var(--accent-gold) !important;
}

/* ===== Upload Area ===== */
.stFileUploader > div {
    border-radius: 10px !important;
    border: 1px dashed rgba(201, 169, 110, 0.12) !important;
    background: rgba(201, 169, 110, 0.01) !important;
    transition: all 0.4s ease !important;
}

.stFileUploader > div:hover {
    border-color: rgba(201, 169, 110, 0.2) !important;
    background: rgba(201, 169, 110, 0.03) !important;
}

/* ===== Button ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-gold), #b8944f) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    padding: 0.8rem 2.5rem !important;
    transition: all 0.4s ease !important;
    color: #0a0a0f !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(201, 169, 110, 0.2) !important;
}

/* ===== Misc ===== */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

.streamlit-expanderHeader {
    border-radius: 10px !important;
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
}

.stPlotlyChart { border-radius: 12px; overflow: hidden; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(201,169,110,0.25); }

.stAudio { border-radius: 10px; overflow: hidden; }

.transcript-text {
    font-size: 0.92rem;
    line-height: 2;
    color: var(--text-secondary);
    padding: 0.5rem 0 0.5rem 1.2rem;
    border-left: 1px solid rgba(201,169,110,0.15);
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Interactive Music-Themed Canvas Background (cursor-reactive)
# ---------------------------------------------------------------------------
components.html("""
<canvas id="musicCanvas" style="
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none;
"></canvas>
<script>
(function() {
    const canvas = document.getElementById('musicCanvas');
    const ctx = canvas.getContext('2d');
    let W, H;
    let mouseX = 0, mouseY = 0;
    let targetX = 0, targetY = 0;

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    // Track cursor across the whole page
    window.addEventListener('mousemove', e => {
        targetX = e.clientX;
        targetY = e.clientY;
    });

    // Also track in parent (Streamlit runs in complex DOM)
    try {
        window.parent.document.addEventListener('mousemove', e => {
            targetX = e.clientX;
            targetY = e.clientY;
        });
    } catch(e) {}

    // Equalizer bars
    const NUM_BARS = 64;
    const bars = [];
    for (let i = 0; i < NUM_BARS; i++) {
        bars.push({
            x: (i / NUM_BARS) * 1.2 - 0.1,
            height: 0,
            targetH: 0,
            phase: Math.random() * Math.PI * 2,
            speed: 0.3 + Math.random() * 0.7
        });
    }

    // Floating music notes / particles
    const PARTICLES = [];
    const SYMBOLS = ['♪', '♫', '♬', '♩', '◦', '·'];
    for (let i = 0; i < 30; i++) {
        PARTICLES.push({
            x: Math.random() * W,
            y: Math.random() * H,
            vx: (Math.random() - 0.5) * 0.3,
            vy: -0.2 - Math.random() * 0.4,
            size: 8 + Math.random() * 14,
            opacity: 0.03 + Math.random() * 0.06,
            symbol: SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)],
            phase: Math.random() * Math.PI * 2
        });
    }

    // Sound wave rings from cursor
    const RINGS = [];
    let ringTimer = 0;

    let time = 0;

    function draw() {
        time += 0.016;
        ctx.clearRect(0, 0, W, H);

        // Smooth cursor follow
        mouseX += (targetX - mouseX) * 0.06;
        mouseY += (targetY - mouseY) * 0.06;

        // -- Draw equalizer bars at bottom --
        const barWidth = W / NUM_BARS;
        for (let i = 0; i < NUM_BARS; i++) {
            const b = bars[i];
            const bx = b.x * W;

            // Cursor proximity influence
            const dx = mouseX - (i / NUM_BARS) * W;
            const proximity = Math.max(0, 1 - Math.abs(dx) / (W * 0.2));

            // Base wave + cursor boost
            b.targetH = (Math.sin(time * b.speed + b.phase) * 0.5 + 0.5) * 25
                       + proximity * 40
                       + Math.sin(time * 1.5 + i * 0.15) * 8;

            b.height += (b.targetH - b.height) * 0.08;

            const alpha = 0.03 + proximity * 0.06;
            const goldR = 201, goldG = 169, goldB = 110;

            ctx.fillStyle = `rgba(${goldR}, ${goldG}, ${goldB}, ${alpha})`;
            const x = i * barWidth;
            ctx.fillRect(x, H - b.height, barWidth - 1, b.height);

            // Mirror on top (very faint)
            ctx.fillStyle = `rgba(${goldR}, ${goldG}, ${goldB}, ${alpha * 0.3})`;
            ctx.fillRect(x, 0, barWidth - 1, b.height * 0.5);
        }

        // -- Floating music note particles --
        for (const p of PARTICLES) {
            p.phase += 0.01;
            const pdx = mouseX - p.x;
            const pdy = mouseY - p.y;
            const dist = Math.sqrt(pdx * pdx + pdy * pdy);

            // Gentle push away from cursor
            if (dist < 200) {
                const force = (200 - dist) / 200 * 0.5;
                p.x -= (pdx / dist) * force;
                p.y -= (pdy / dist) * force;
            }

            p.x += p.vx + Math.sin(p.phase) * 0.3;
            p.y += p.vy;

            // Wrap around
            if (p.y < -30) { p.y = H + 30; p.x = Math.random() * W; }
            if (p.x < -30) p.x = W + 30;
            if (p.x > W + 30) p.x = -30;

            // Cursor proximity glow
            const glowBoost = dist < 250 ? (250 - dist) / 250 * 0.08 : 0;

            ctx.font = `${p.size}px Inter, sans-serif`;
            ctx.fillStyle = `rgba(201, 169, 110, ${p.opacity + glowBoost})`;
            ctx.fillText(p.symbol, p.x, p.y);
        }

        // -- Sound wave rings from cursor --
        ringTimer += 0.016;
        if (ringTimer > 1.2 && (targetX > 0 || targetY > 0)) {
            ringTimer = 0;
            RINGS.push({ x: mouseX, y: mouseY, r: 0, opacity: 0.06 });
        }

        for (let i = RINGS.length - 1; i >= 0; i--) {
            const ring = RINGS[i];
            ring.r += 1.5;
            ring.opacity -= 0.0004;

            if (ring.opacity <= 0) { RINGS.splice(i, 1); continue; }

            ctx.beginPath();
            ctx.arc(ring.x, ring.y, ring.r, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(201, 169, 110, ${ring.opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        // -- Subtle horizontal sound wave line --
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(201, 169, 110, 0.025)';
        ctx.lineWidth = 1;
        const waveY = H * 0.5;
        for (let x = 0; x < W; x += 3) {
            const dx = mouseX - x;
            const proximity = Math.max(0, 1 - Math.abs(dx) / (W * 0.25));
            const amp = 3 + proximity * 30;
            const y = waveY + Math.sin(x * 0.01 + time * 2) * amp
                     + Math.sin(x * 0.005 + time * 0.8) * amp * 0.5;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        requestAnimationFrame(draw);
    }

    draw();
})();
</script>
""", height=0)

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
# Emotion Config
# ---------------------------------------------------------------------------
ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise", "excitement"]

EMOTION_COLORS = {
    "joy": "#e8c885",
    "sadness": "#6688aa",
    "anger": "#c25050",
    "fear": "#8866aa",
    "neutral": "#6b6b80",
    "surprise": "#cc8844",
    "disgust": "#668855",
    "excitement": "#cc6688",
}

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😡",
    "fear": "😨", "neutral": "😐", "surprise": "😲",
    "disgust": "🤢", "excitement": "🤩",
}

NEUTRAL_DAMPEN = 0.55
FEAR_DAMPEN = 0.50
MIN_EMOTION_CONF = 0.15
MERGE_WINDOW_SEC = 12.0

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _check_excitement(scores: dict[str, float]) -> tuple[bool, float]:
    joy = scores.get("joy", 0)
    surprise = scores.get("surprise", 0)
    combined = joy + surprise
    dominant = max(scores, key=scores.get)
    excited = (
        (dominant == "joy" and joy >= 0.25)
        or (dominant == "surprise" and surprise >= 0.35)
        or (combined >= 0.35)
    )
    if excited:
        conf = round(max(combined, joy, surprise), 3)
        return True, max(conf, 0.20)
    return False, 0.0


def merge_segments(raw_segments: list[dict], window_sec: float = MERGE_WINDOW_SEC) -> list[dict]:
    if not raw_segments:
        return []
    merged = []
    buf_texts: list[str] = []
    buf_start: float = raw_segments[0]["start"]
    buf_end: float = raw_segments[0]["end"]
    for seg in raw_segments:
        text = seg["text"].strip()
        if not text:
            continue
        if not buf_texts:
            buf_start = seg["start"]
            buf_end = seg["end"]
            buf_texts.append(text)
            continue
        current_len = buf_end - buf_start
        if current_len >= window_sec:
            merged.append({"start": buf_start, "end": buf_end, "text": " ".join(buf_texts)})
            buf_texts = [text]
            buf_start = seg["start"]
            buf_end = seg["end"]
        else:
            buf_texts.append(text)
            buf_end = seg["end"]
    if buf_texts:
        merged.append({"start": buf_start, "end": buf_end, "text": " ".join(buf_texts)})
    return merged


def classify_emotion(text: str, emotion_clf) -> dict:
    tokenizer = emotion_clf.tokenizer
    tokens = tokenizer.encode(text, truncation=True, max_length=512)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    emotions = emotion_clf(text)[0]
    score_dict = {e["label"]: e["score"] for e in emotions}
    if "neutral" in score_dict:
        score_dict["neutral"] *= NEUTRAL_DAMPEN
    if "fear" in score_dict:
        score_dict["fear"] *= FEAR_DAMPEN
    total = sum(score_dict.values())
    if total > 0:
        score_dict = {k: v / total for k, v in score_dict.items()}
    is_excited, exc_conf = _check_excitement(score_dict)
    if is_excited:
        emo_label = "excitement"
        confidence = exc_conf
    else:
        dominant = max(score_dict, key=score_dict.get)
        confidence = score_dict[dominant]
        if dominant != "neutral" and confidence < MIN_EMOTION_CONF:
            emo_label = "neutral"
            confidence = score_dict.get("neutral", confidence)
        else:
            emo_label = dominant
    return {
        "emotion": emo_label,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 4) for k, v in score_dict.items()},
    }


def detect_change_points(segments: list[dict]) -> list[dict]:
    changes = []
    for i in range(1, len(segments)):
        if segments[i]["emotion"] != segments[i - 1]["emotion"]:
            changes.append({
                "time": segments[i]["start"],
                "time_fmt": _fmt(segments[i]["start"]),
                "from": segments[i - 1]["emotion"],
                "to": segments[i]["emotion"],
            })
    return changes


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Voice Analysis</div>
    <div class="hero-title">🎙️ Sentiment <span>Analyzer</span></div>
    <div class="gold-line"></div>
    <div class="hero-subtitle">
        Upload an audio file or record from your microphone —
        powered by Whisper ASR and emotion classification.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input: Upload OR Record
# ---------------------------------------------------------------------------
st.markdown('<div class="glass-card"><div class="card-title">🎵 Audio Input</div>', unsafe_allow_html=True)
tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎤 Record from Mic"])

audio_bytes = None
audio_suffix = ".wav"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "m4a", "ogg"]
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        audio_bytes = uploaded_file.getvalue()
        audio_suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"

with tab_mic:
    st.caption("Click the microphone button below to record your voice.")
    recorded = st.audio_input("Record audio")
    if recorded:
        audio_bytes = recorded.getvalue()
        audio_suffix = ".wav"

st.markdown('</div>', unsafe_allow_html=True)

if audio_bytes is not None:
    if st.button("⚡ Analyze", type="primary"):
        try:
            with st.spinner("Transcribing and Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    try:
                        result = asr_model.transcribe(tmp_path)
                    except RuntimeError as re:
                        if "reshape" in str(re) or "0 elements" in str(re):
                            st.error(
                                "⚠️ The audio is too short or contains no audible speech. "
                                "Please upload a longer recording (at least 1-2 seconds of clear speech)."
                            )
                            st.stop()
                        raise  # re-raise if it's a different RuntimeError

                    raw_segments = result.get("segments", [])
                    full_text = result.get("text", "")

                    if not raw_segments:
                        st.error("No speech detected. Try a clearer recording.")
                        st.stop()

                    duration = raw_segments[-1]["end"]
                    merged = merge_segments(raw_segments, window_sec=MERGE_WINDOW_SEC)

                    segments: list[dict] = []
                    emo_dur: dict[str, float] = {}

                    for win in merged:
                        emo = classify_emotion(win["text"], emotion_clf)
                        seg_dur = win["end"] - win["start"]
                        emo_dur[emo["emotion"]] = emo_dur.get(emo["emotion"], 0) + seg_dur
                        segments.append({"start": win["start"], "end": win["end"], "text": win["text"], **emo})

                    fine_segments: list[dict] = []
                    for seg in raw_segments:
                        text = seg["text"].strip()
                        if not text:
                            continue
                        idx = raw_segments.index(seg)
                        context_texts = []
                        for j in range(max(0, idx - 2), min(len(raw_segments), idx + 3)):
                            t = raw_segments[j]["text"].strip()
                            if t:
                                context_texts.append(t)
                        context = " ".join(context_texts)
                        emo = classify_emotion(context, emotion_clf)
                        fine_segments.append({"start": seg["start"], "end": seg["end"], "text": text, **emo})

                    changes = detect_change_points(segments)

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # ================================================================
            # RESULTS
            # ================================================================

            st.markdown("""
            <div class="status-badge">
                ✓ Analysis Complete
            </div>
            """, unsafe_allow_html=True)

            # ---- Dominant Emotion ----
            if emo_dur:
                overall_dominant = max(emo_dur, key=emo_dur.get)
                emoji = EMOTION_EMOJI.get(overall_dominant, "😶")
                st.markdown(f"""
                <div class="emotion-hero">
                    <div class="emotion-emoji-big">{emoji}</div>
                    <div class="emotion-label-big">{overall_dominant}</div>
                    <div class="emotion-sublabel">Dominant Emotion</div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Metrics ----
            word_count = len(full_text.split())
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-icon">⏱</div>
                    <div class="metric-value">{duration:.1f}s</div>
                    <div class="metric-label">Duration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">✎</div>
                    <div class="metric-value">{word_count}</div>
                    <div class="metric-label">Words</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">▧</div>
                    <div class="metric-value">{len(segments)}</div>
                    <div class="metric-label">Segments</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">↝</div>
                    <div class="metric-value">{len(changes)}</div>
                    <div class="metric-label">Shifts</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ---- Timeline Chart ----
            st.markdown('<div class="glass-card"><div class="card-title">🕐 Emotion Timeline</div>', unsafe_allow_html=True)
            st.caption("How emotions change throughout the audio.")

            timeline_rows = []
            for s in segments:
                mid = (s["start"] + s["end"]) / 2
                timeline_rows.append({
                    "Time (s)": round(mid, 1),
                    "Timestamp": _fmt(mid),
                    "Emotion": s["emotion"].capitalize(),
                    "Confidence": s["confidence"],
                    "Text": s["text"][:80] + ("…" if len(s["text"]) > 80 else ""),
                })
            timeline_df = pd.DataFrame(timeline_rows)

            all_emotions_cap = [e.capitalize() for e in ALL_EMOTIONS]
            emo_y_map = {e: i for i, e in enumerate(all_emotions_cap)}
            timeline_df["Emotion_Y"] = timeline_df["Emotion"].map(emo_y_map)

            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=timeline_df["Time (s)"],
                y=timeline_df["Emotion_Y"],
                mode="lines",
                line=dict(color="rgba(201,169,110,0.15)", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

            for emo_name in all_emotions_cap:
                mask = timeline_df["Emotion"] == emo_name
                subset = timeline_df[mask]
                color = EMOTION_COLORS.get(emo_name.lower(), "#6b6b80")

                if len(subset) > 0:
                    fig_timeline.add_trace(go.Scatter(
                        x=subset["Time (s)"],
                        y=subset["Emotion_Y"],
                        mode="markers",
                        marker=dict(size=12, color=color, line=dict(width=1, color="rgba(255,255,255,0.15)"), symbol="circle"),
                        name=f"{EMOTION_EMOJI.get(emo_name.lower(), '😶')} {emo_name}",
                        text=subset["Text"],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Emotion: %{customdata[1]}<br>"
                            "Confidence: %{customdata[2]:.0%}<br>"
                            "Text: %{text}<extra></extra>"
                        ),
                        customdata=list(zip(subset["Timestamp"], subset["Emotion"], subset["Confidence"])),
                    ))
                else:
                    fig_timeline.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(size=10, color=color),
                        name=f"{EMOTION_EMOJI.get(emo_name.lower(), '😶')} {emo_name}",
                        showlegend=True,
                    ))

            fig_timeline.update_layout(
                yaxis=dict(
                    tickvals=list(emo_y_map.values()),
                    ticktext=[f"{EMOTION_EMOJI.get(e.lower(), '')} {e}" for e in emo_y_map.keys()],
                    title="",
                    gridcolor="rgba(255,255,255,0.02)",
                    range=[-0.5, len(all_emotions_cap) - 0.5],
                ),
                xaxis=dict(title="Time (seconds)", gridcolor="rgba(255,255,255,0.02)"),
                height=400,
                margin=dict(l=120, r=20, t=20, b=50),
                legend=dict(orientation="h", y=-0.25, font=dict(size=10, color="#6b6b80")),
                plot_bgcolor="rgba(8,8,12,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#6b6b80", family="Inter", size=11),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Change Points ----
            st.markdown('<div class="glass-card"><div class="card-title">↝ Emotion Shifts</div>', unsafe_allow_html=True)
            if changes:
                change_html = ""
                for c in changes:
                    from_emoji = EMOTION_EMOJI.get(c['from'], '😶')
                    to_emoji = EMOTION_EMOJI.get(c['to'], '😶')
                    change_html += f"""
                    <div class="change-point">
                        <div class="change-time">{c['time_fmt']}</div>
                        <div>{from_emoji} {c['from'].capitalize()}</div>
                        <div class="change-arrow">→</div>
                        <div>{to_emoji} {c['to'].capitalize()}</div>
                    </div>
                    """
                st.markdown(change_html, unsafe_allow_html=True)
            else:
                st.info("No emotion shifts detected — consistent emotion throughout.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ---- Distribution ----
            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.markdown('<div class="glass-card"><div class="card-title">◕ Distribution</div>', unsafe_allow_html=True)
                pie_data = [{"Emotion": e.capitalize(), "Duration (s)": round(emo_dur.get(e, 0), 1)} for e in ALL_EMOTIONS]
                fig_pie = px.pie(
                    pie_data, values="Duration (s)", names="Emotion",
                    color="Emotion",
                    color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()},
                    hole=0.6,
                )
                fig_pie.update_layout(
                    height=360, margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b6b80", family="Inter", size=10),
                    legend=dict(font=dict(size=10)),
                    showlegend=True,
                )
                fig_pie.update_traces(
                    textinfo="percent",
                    textfont_size=9,
                    marker=dict(line=dict(color="rgba(8,8,12,0.8)", width=2)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_bar:
                st.markdown('<div class="glass-card"><div class="card-title">▥ Scores / Segment</div>', unsafe_allow_html=True)
                bar_rows = []
                for i, s in enumerate(segments):
                    for emo_key in ALL_EMOTIONS:
                        bar_rows.append({
                            "Segment": f"{_fmt(s['start'])}–{_fmt(s['end'])}",
                            "Emotion": emo_key.capitalize(),
                            "Score": s["scores"].get(emo_key, 0.0),
                        })
                bar_df = pd.DataFrame(bar_rows)
                fig_bar = px.bar(
                    bar_df, x="Segment", y="Score", color="Emotion",
                    color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()},
                    barmode="stack",
                )
                fig_bar.update_layout(
                    height=360, margin=dict(t=10, b=10),
                    xaxis=dict(title="", gridcolor="rgba(255,255,255,0.02)"),
                    yaxis=dict(title="", gridcolor="rgba(255,255,255,0.02)"),
                    plot_bgcolor="rgba(8,8,12,0.9)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b6b80", family="Inter", size=10),
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ---- Emotion Pills ----
            st.markdown('<div class="glass-card"><div class="card-title">🎭 Detected Emotions</div>', unsafe_allow_html=True)
            pills_html = '<div class="emotion-pills">'
            for e in ALL_EMOTIONS:
                dur_val = emo_dur.get(e, 0)
                emoji = EMOTION_EMOJI.get(e, "😶")
                color = EMOTION_COLORS.get(e, "#6b6b80")
                opacity = "1" if dur_val > 0 else "0.3"
                pills_html += f"""
                <div class="emotion-pill" style="opacity: {opacity};">
                    <span>{emoji}</span>
                    <span style="color: {color}; font-weight: 500;">{e.capitalize()}</span>
                    <span class="pill-dur">{dur_val:.1f}s</span>
                </div>
                """
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ---- Segment Table ----
            st.markdown('<div class="glass-card"><div class="card-title">📋 Emotion Log</div>', unsafe_allow_html=True)
            table_rows = []
            for s in fine_segments:
                emoji = EMOTION_EMOJI.get(s["emotion"], "😶")
                table_rows.append({
                    "Start": _fmt(s["start"]),
                    "End": _fmt(s["end"]),
                    "Emotion": f"{emoji} {s['emotion'].capitalize()}",
                    "Confidence": f"{s['confidence']:.0%}",
                    "Text": s["text"],
                })
            table_df = pd.DataFrame(table_rows)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ---- Transcript ----
            st.markdown('<div class="glass-card"><div class="card-title">📝 Transcript</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-text">{full_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Raw JSON ----
            with st.expander("🗂️ Raw JSON"):
                st.json({
                    "duration": round(duration, 2),
                    "segments": segments,
                    "fine_segments": fine_segments,
                    "change_points": changes,
                    "distribution": {k: round(v, 2) for k, v in emo_dur.items()},
                })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
