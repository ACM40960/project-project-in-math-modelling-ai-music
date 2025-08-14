import argparse, os
from pydub import AudioSegment
from hybrid_infer import hybrid_predict
from part2_musicgen.longgen import generate_long_from_emotion

def add_fades_and_export(wav_path, fade_in_ms=400, fade_out_ms=1200, export_mp3=True):
    audio = AudioSegment.from_wav(wav_path).fade_in(fade_in_ms).fade_out(fade_out_ms)
    audio.export(wav_path, format="wav")
    if export_mp3:
        mp3_path = wav_path[:-4] + ".mp3"
        audio.export(mp3_path, format="mp3", bitrate="192k")
        return mp3_path
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="raw text; if provided, we'll detect emotion")
    ap.add_argument("--emotion", help="manual emotion override (joy/sadness/anger/fear/love/gratitude/calm/neutral)")
    ap.add_argument("--total", type=int, default=90, help="target length in seconds")
    ap.add_argument("--chunk", type=int, default=8, help="chunk length (VRAM-safe: 6–10)")
    ap.add_argument("--overlap", type=float, default=1.0, help="crossfade length seconds")
    ap.add_argument("--prompt", default=None, help="override the generated prompt")
    ap.add_argument("--outfile", default="outputs/track.wav")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # pick an emotion
    emotion = args.emotion
    prompt = args.prompt
    if args.text and not emotion:
        r = hybrid_predict(args.text)
        emotion = r["chosen_emotion_coarse"]
        # confidence-aware prompt tweak
        if prompt is None and r["chosen_prob"] < 0.55 and emotion in ("sadness","fear"):
            prompt = "slow minor-key piano and soft pads, sparse and intimate"
        print(f"Detected → coarse={emotion}, fine={r['chosen_emotion_fine']}, "
              f"distress={r['distress']}({r['distress_conf']})")

    if not emotion:
        raise SystemExit("Provide --text to detect emotion or --emotion to set it manually.")

    # generate long audio via chunking
    wav_path, used_prompt = generate_long_from_emotion(
        emotion, total_sec=args.total, chunk_sec=args.chunk, overlap_sec=args.overlap,
        outfile=args.outfile, override_prompt=prompt, seed=args.seed
    )
    print(f"\n✅ Saved WAV: {wav_path}\n📝 Prompt used: {used_prompt}")

    # polish + MP3
    try:
        mp3_path = add_fades_and_export(wav_path)
        print(f"🎵 Also wrote MP3: {mp3_path}")
    except Exception as e:
        print("MP3 export skipped:", e)

if __name__ == "__main__":
    main()
