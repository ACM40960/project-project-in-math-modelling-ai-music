# part2_musicgen/pipeline.py
import argparse
from .longgen import generate_long_from_emotion

def main():
    # CLI arguments for generating a track
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion", required=True, help="joy/sadness/anger/fear/love/gratitude/calm/neutral")
    ap.add_argument("--total", type=int, default=90, help="target length in seconds")
    ap.add_argument("--chunk", type=int, default=8, help="chunk size in seconds (memory-friendly)")
    ap.add_argument("--overlap", type=float, default=1.0, help="crossfade duration in seconds")
    ap.add_argument("--outfile", default="outputs/track.wav", help="output WAV path")
    ap.add_argument("--prompt", default=None, help="optional custom text prompt")
    ap.add_argument("--seed", type=int, default=None, help="random seed for repeatability")
    ap.add_argument("--variant", type=int, default=None, help="specific variant index")
    ap.add_argument("--mode", choices=["variety","adhere","clean"], default=None, help="param tweak profile")
    args = ap.parse_args()

    # Run generation
    path, prompt = generate_long_from_emotion(
        emotion=args.emotion,
        total_sec=args.total,
        chunk_sec=args.chunk,
        overlap_sec=args.overlap,
        outfile=args.outfile,
        override_prompt=args.prompt,
        seed=args.seed,
        variant=args.variant,
        mode=args.mode,
    )
    print(f" Wrote {path}\n Prompt: {prompt}")

if __name__ == "__main__":
    main()
