# app.py
import os, io, time, json, warnings, random
import torch
import streamlit as st
from audiocraft.models import MusicGen
from pydub import AudioSegment
from pydub.utils import which
from datetime import datetime

# target emotion 
REDIRECT = {
    "soothe": {
        "fear":"calm", "sadness":"calm", "anger":"calm", "grief":"calm",
        "disgust":"calm", "nervousness":"calm", "disappointment":"calm",
        "neutral":"calm",
        "joy":"joy", "gratitude":"gratitude", "love":"love", "calm":"calm",
    },
    "uplift": {
        "fear":"gratitude", "sadness":"gratitude", "anger":"love",
        "neutral":"joy", "calm":"joy",
        "joy":"joy", "gratitude":"gratitude", "love":"love",
    },
}
def map_goal(emotion: str, goal: str) -> str:
    if goal == "express":
        return emotion
    return REDIRECT.get(goal, {}).get(emotion, emotion)

# Streamlit basics
st.set_page_config(page_title="MoodMelody", layout="wide")
warnings.filterwarnings("ignore", message=".*Triton.*")
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
# Local modules
from hybrid_infer import hybrid_predict
from part2_musicgen.longgen import generate_long_from_emotion

# FFmpeg (optional MP3)
AudioSegment.converter = which("ffmpeg") or AudioSegment.converter
if not AudioSegment.converter:
    st.warning("FFmpeg not found — MP3 export will be skipped.")

# Cache MusicGen
@st.cache_resource
def get_musicgen():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MusicGen.get_pretrained("facebook/musicgen-small", device=dev)

# Session state
for k, v in {
    "detected": None, "emotion": None, "text": "",
    "last_wav": None, "last_mp3": None, "last_prompt": None,
    "history": [], "busy": False,
}.items():
    st.session_state.setdefault(k, v)

# Style and Background
def inject_css(base_css: str, extra_css: str = ""):
    st.markdown(f"<style>{base_css}\n{extra_css}</style>", unsafe_allow_html=True)

BASE_CSS = """
.block-container { padding-top: 3rem; padding-left: 2rem; padding-right: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
h1 { font-weight: 800; margin-bottom: .25rem; }
h2 { font-weight: 700; }
hr {border:none; border-top:1px solid rgba(255,255,255,.08); margin:1rem 0;}
.card { padding: 1rem; border-radius: 16px; background: rgba(255,255,255,.03);
        border: 1px solid rgba( ); }
.stButton>button { border-radius: 12px; padding: .55rem 1rem; font-weight: 700; }
.stSlider, .stNumberInput, .stTextInput { border-radius: 12px; }
.badge { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.80rem;
         border:1px solid rgba(255,255,255,.15); }
.badge.safe { background:#0f5132; color:#d1fae5; border-color:#14532d; }
.badge.distress { background:#512a2a; color:#fee2e2; border-color:#7f1d1d; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
"""
BLOBS_BG = """
[data-testid="stAppViewContainer"]::before {
  content:"";
  position: fixed; inset: -20% -20% -20% -20%;
  background:
    radial-gradient(40% 40% at 20% 30%, rgba(20,184,166,.14), transparent 70%),
    radial-gradient(35% 35% at 80% 20%, rgba(99,102,241,.13), transparent 70%),
    radial-gradient(45% 45% at 60% 75%, rgba(56,189,248,.10), transparent 70%);
  filter: blur(40px);
  z-index: -1; opacity: .9;
  animation: blobs 40s ease-in-out infinite alternate;
}
@keyframes blobs {
  0%   { transform: translate3d(0,0,0) scale(1); }
  50%  { transform: translate3d(-2%,1%,0) scale(1.03); }
  100% { transform: translate3d(2%,-1%,0) scale(1.01); }
}
"""
QUOTE_CSS = """
.quote-card {
  margin:.5rem 0 0.75rem 0; padding: 0.9rem 1rem; border-radius: 14px;
  background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.06);
  font-size: 1.02rem; line-height: 1.55; letter-spacing:.2px;
}
.quote-fade { animation: qfade 0.6s ease; }
@keyframes qfade { from { opacity:0; transform: translateY(2px);} to {opacity:1; transform:none;} }
"""
inject_css(BASE_CSS, BLOBS_BG + QUOTE_CSS)

# Quotes
QUOTES = {
    "calm": ["Breathe in peace, breathe out tension.","Let this moment be enough.",
             "You are safe here. Slow and steady.","Softness is strength too.","It’s okay to pause."],
    "gratitude": ["Let small joys be loud.","Remember what stayed when you needed it.",
                  "Thank the quiet moments.","Gratitude turns what we have into enough.","Notice the gentle good."],
    "joy": ["Let light in where it’s kind.","A little spark can be a sunrise.",
            "Joy doesn’t have to shout.","Smile with your shoulders; breathe with your chest.","Let your heart feel lighter."],
    "love": ["Be kind to the person you are becoming.","Offer yourself the same warmth you give others.",
             "You deserve gentleness.","Let care be your compass.","Hold yourself with patience."],
    "sadness": ["It’s okay to feel this. You’re not alone.","Tears water new beginnings.",
                "Softly, one breath at a time.","Let the weight rest for a while.","You can take up space with your feelings."],
    "fear": ["Name the fear; it shrinks.","You can move slowly and still arrive.",
             "Safety can be a rhythm.","Anchor to your breath.","One calm note after another."],
    "anger": ["Let heat meet breath and soften.","Ground through the soles of your feet.",
              "You can choose when to speak and when to rest.","Strength can be quiet.","Let the storm pass safely."],
    "neutral": ["Return to center.","Notice, without judgment.","Quiet is also music.",
                "Be where your feet are.","Begin again, gently."],
}
def quotes_for(emotion: str):
    return QUOTES.get((emotion or "calm").lower(), QUOTES["calm"])

# header
st.markdown("""
<div style="display:flex;align-items:center;gap:.6rem;margin-top:-.3rem">
  <svg width="28" height="28" viewBox="0 0 24 24" fill="#14b8a6" xmlns="http://www.w3.org/2000/svg">
    <path d="M9 3v12.5a3.5 3.5 0 1 1-2-3.146V3h2zm8 2v7.5a3.5 3.5 0 1 1-2-3.146V5h2z"/>
  </svg>
  <div style="font-weight:800;font-size:1.75rem;line-height:1.2;">MoodMelody</div>
</div>
""", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-top:0.2rem;font-size:1.05rem;color:rgba(255,255,255,0.85);'>"
    "When words fail, music speaks and heals. Choose <em>express</em> to match your mood, "
    "or <em>soothe/uplift</em> to gently guide it."
    "</p>", unsafe_allow_html=True
)

tab_create, tab_history, tab_about = st.tabs(["Create", "History", "About"])

with tab_create:
    left, right = st.columns([1.5, 1.15], gap="large")

    # emotion detection part
    with left:
        st.markdown("### Detect emotion")
        st.radio("Source", ["Detect from text", "Manual"], horizontal=True, key="mode_src")

        if st.session_state.mode_src == "Detect from text":
            st.session_state.text = st.text_area(
                "Describe how you feel:",
                value=st.session_state.text or "",
                placeholder="e.g., I feel so hopeless and tired of everything.",
                height=140
            )
            if st.button("Detect", key="detect_btn"):
                txt = (st.session_state.text or "").strip()
                if not txt:
                    st.warning("Please enter some text.")
                else:
                    with st.spinner("Detecting emotion…"):
                        res = hybrid_predict(txt)
                    st.session_state.detected = res
                    st.session_state.emotion = res["chosen_emotion_coarse"]
        else:
            st.session_state.emotion = st.selectbox(
                "Pick an emotion", ["joy","gratitude","love","calm","neutral","sadness","fear","anger"], index=1
            )
            st.session_state.text = st.text_area("Optional note", value=st.session_state.text or "", height=100)

        det = st.session_state.get("detected")
        if det:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Detected:** {det['chosen_emotion_coarse']}  ·  Fine: {det['chosen_emotion_fine']}")
            badge = "distress" if det["distress"] else "safe"
            label = "Distress" if det["distress"] else "Safe"
            st.markdown(f'<span class="badge {badge}">{label} · {det["distress_conf"]:.2f}</span>',
                        unsafe_allow_html=True)

            # transparent bar chart
            try:
                import matplotlib.pyplot as plt, matplotlib as mpl
                mpl.rcParams.update({
                    "figure.facecolor": "none","axes.facecolor":"none","axes.edgecolor":"#9CA3AF",
                    "xtick.color":"#E5E7EB","ytick.color":"#E5E7EB","text.color":"#E5E7EB","grid.color":"#FFFFFF",
                })
                labs = [e for e,_ in det["top_emotions"]][:5]
                vals = [p for _,p in det["top_emotions"]][:5]
                fig, ax = plt.subplots()
                fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                ax.bar(labs, vals, color="#93C5FD", edgecolor="none")
                ax.set_ylim(0,1); ax.set_title("Top emotions", color="#E5E7EB")
                for s in ["top","right"]: ax.spines[s].set_visible(False)
                ax.spines["left"].set_color("#9CA3AF"); ax.spines["bottom"].set_color("#9CA3AF")
                ax.grid(axis="y", alpha=0.15); ax.tick_params(axis="x", rotation=0)
                st.pyplot(fig, use_container_width=True)
            except Exception:
                st.json(det["top_emotions"][:5])
            st.markdown("</div>", unsafe_allow_html=True)

    # music generation part
    with right:
        st.markdown("### Generate music")
        custom_prompt = st.text_input("Override prompt (optional)", "")
        total = st.slider("Total length (sec)", 30, 180, 90, step=10)
        cols = st.columns(3)
        chunk   = cols[0].slider("Chunk", 6, 12, 8, step=1)
        overlap = cols[1].slider("Overlap", 0.0, 2.0, 1.0, step=0.1)
        seed    = cols[2].number_input("Seed", min_value=0, max_value=999999, value=1234, step=1)

        goal = st.radio("Music goal", ["express", "soothe", "uplift"], index=1, horizontal=True)
        cols2 = st.columns([1,1])
        tone_mode = cols2[0].selectbox("Tone tweak", ["default","variety","adhere","clean"], index=0)
        variant   = cols2[1].number_input("Variant", min_value=0, value=0, step=1)

        if st.button("Generate", key="gen_btn"):
            if st.session_state.busy:
                st.info("A track is already rendering — please wait.")
                st.stop()
            st.session_state.busy = True
            try:
                detected_emo = st.session_state.emotion
                if not detected_emo:
                    st.warning("Please detect or pick an emotion.")
                    st.stop()

                gen_emo = map_goal(detected_emo, goal)
                if gen_emo != detected_emo:
                    st.info(f"Detected '{detected_emo}' → generating '{gen_emo}' (goal: {goal}).")

                prompt_override = (custom_prompt.strip() or None)
                det = st.session_state.get("detected")
                if det and not prompt_override and det["chosen_prob"] < 0.55 and gen_emo in ("sadness","fear"):
                    prompt_override = "slow minor-key piano and soft pads, sparse and intimate"

                # progress UI: bar + ETA + rotating quote
                def fmt_secs(s: float) -> str:
                    s = max(0, float(s)); m = int(s // 60); r = int(round(s - 60*m))
                    return f"{m}:{r:02d}" if m else f"{r}s"

                bar = st.progress(0)
                eta_box = st.empty()
                quote_box = st.empty()
                quotes = quotes_for(gen_emo)

                def meter_cb(i_done, n_total, chunk_sec, active_sec, eta_sec):
                    pct = int(100 * i_done / max(1, n_total))
                    bar.progress(pct)
                    eta_box.markdown(
                        f"**Rendering chunk {i_done}/{n_total}** · "
                        f"chunk {chunk_sec:.1f}s · active {active_sec:.1f}s · "
                        f"ETA ~{fmt_secs(eta_sec)}"
                    )
                    if quotes:
                        quote = random.choice(quotes)
                        quote_box.markdown(
                            f"<div class='quote-card quote-fade'>“{quote}”</div>", unsafe_allow_html=True
                        )

                # render
                os.makedirs("outputs", exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = f"outputs/{gen_emo}_{stamp}.wav"

                with st.spinner("Generating audio…"):
                    model = get_musicgen()
                    wav_path, used_prompt = generate_long_from_emotion(
                        emotion=gen_emo, total_sec=int(total), chunk_sec=int(chunk),
                        overlap_sec=float(overlap), outfile=out,
                        override_prompt=prompt_override, seed=int(seed),
                        variant=int(variant), mode=(None if tone_mode=="default" else tone_mode),
                        model=model, meter_cb=meter_cb,
                    )
                bar.empty(); eta_box.empty(); quote_box.empty()

                # fades + optional mp3
                mp3 = None
                try:
                    audio = AudioSegment.from_wav(wav_path).fade_in(400).fade_out(1200)
                    audio.export(wav_path, format="wav")
                    if AudioSegment.converter:
                        mp3 = wav_path[:-4] + ".mp3"
                        audio.export(mp3, format="mp3", bitrate="192k")
                except Exception as e:
                    st.info(f"MP3 export skipped: {e}")

                st.success("Track ready")
                st.write("**Prompt used:**", used_prompt)
                with open(wav_path, "rb") as f:
                    wav_bytes = f.read()
                st.audio(wav_bytes, format="audio/wav")
                st.download_button("Download WAV", data=wav_bytes,
                                   file_name=os.path.basename(wav_path), mime="audio/wav")
                if mp3 and os.path.exists(mp3):
                    with open(mp3, "rb") as f:
                        mp3_bytes = f.read()
                    st.download_button("Download MP3", data=mp3_bytes,
                                       file_name=os.path.basename(mp3), mime="audio/mpeg")

                hist = st.session_state.get("history", [])
                hist.append({
                    "path": wav_path, "mp3": mp3,
                    "detected": detected_emo, "generated": gen_emo,
                    "prompt": used_prompt, "seed": int(seed),
                    "mode": ("default" if tone_mode=="default" else tone_mode),
                    "variant": int(variant), "secs": int(total), "ts": stamp,
                })
                st.session_state["history"] = hist[-6:]
            except RuntimeError as e:
                st.error("Generation failed. Try a shorter chunk or different seed.")
                st.exception(e)
            finally:
                st.session_state.busy = False

with tab_history:
    st.markdown("### Recent generations")
    hist = st.session_state.get("history", [])
    if not hist:
        st.caption("No tracks yet.")
    else:
        cols = st.columns(2)
        for i, item in enumerate(reversed(hist)):
            c = cols[i % 2]
            with c:
                det_lab = item.get("detected", "?")
                gen_lab = item.get("generated", det_lab)
                st.markdown(f"**{det_lab} → {gen_lab}** · {item['secs']}s  \n"
                            f"_seed {item['seed']} · mode {item['mode']} · var {item['variant']}_")
                try:
                    with open(item["path"], "rb") as f: b = f.read()
                    st.audio(b, format="audio/wav")
                    st.download_button("Download WAV", data=b,
                        file_name=os.path.basename(item["path"]), mime="audio/wav", key=f"dlw_{i}")
                except Exception:
                    st.caption("(missing file)")
                if item.get("mp3") and item["mp3"] and os.path.exists(item["mp3"]):
                    with open(item["mp3"], "rb") as f: mb = f.read()
                    st.download_button("Download MP3", data=mb,
                        file_name=os.path.basename(item["mp3"]), mime="audio/mpeg", key=f"dlm_{i}")

with tab_about:
    st.markdown("""
### About MoodMelody

**Why mental health matters**  
Mental health shapes how we think, feel, and act. It affects our ability to handle stress, relate to others, and make choices. Caring for it builds resilience, creativity, and healthier relationships.

**How music helps**  
Music can calm the nervous system, lower stress hormones, and steady breathing and heart rate. Gentle, steady patterns can reduce anxiety; brighter harmonies can lift mood and restore hope. Sometimes simply listening with intention is enough to change how a moment feels.

**How this app works**  
1. Share how you feel (or pick an emotion).  
2. AI detects your emotional state using language models trained on emotional datasets.  
3. We generate music that either **expresses** your feeling or gently **guides** it toward calm or positivity (*soothe/uplift*).  
4. Listen, download, and revisit past tracks as part of your self-care ritual.

**Tips for a better experience**  
• Use headphones and a comfortable volume.  
• Try slow breathing with the beat (e.g., 4 counts in, 6 counts out).  
• Journal a sentence about how you feel before and after listening.  
• If a track doesn’t fit, try another seed or switch between *express / soothe / uplift*.

**Privacy & safety**  
Your text and audio stay on your device when running locally; nothing is stored unless you save it.  
MoodMelody is a wellbeing tool, **not** medical advice. If you’re in crisis or feel unsafe, please reach out to local emergency services or a trusted professional.
""")
