# tools/estimate_cost.py
import os, json, argparse, time
from datetime import datetime
from pathlib import Path

import torch
from audiocraft.models import MusicGen
from part2_musicgen.longgen import generate_long_from_emotion

# Rough $/minute (edit to your provider’s current rates)
RATES_PER_MIN = {
    "modal.L4":          0.12,
    "modal.A10G":        0.102,
    "modal.A100-40GB":   0.42,
    "modal.H100-80GB":   1.32,
    "replicate.T4":      0.012,
    "replicate.L40S":    0.18,
    "runpod.4090_vm":    0.0057,  # $/min (hourly ÷ 60); excludes idle
}

def fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    m = int(s // 60)
    r = int(round(s - 60*m))
    return f"{m}:{r:02d}" if m else f"{r}s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion", required=True)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--mode", default=None, help="variety|adhere|clean|None")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--overhead-sec", type=float, default=0.0, help="cold start / HTTP overhead seconds")
    ap.add_argument("--logdir", default="outputs/estimates")
    args = ap.parse_args()

    # timing bucket
    per_chunk = []
    active_so_far = 0.0

    def meter_cb(i_done, n_total, chunk_sec, active_sec, eta_sec):
        nonlocal active_so_far
        per_chunk.append(float(chunk_sec))
        active_so_far = float(active_sec)
        print(f"[meter] chunk {i_done}/{n_total}: {chunk_sec:.2f}s · active {active_sec:.2f}s · ETA ~{fmt_secs(eta_sec)}")

    # cache/load model once (local run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)

    # ensure outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.outfile or f"outputs/{args.emotion}_{stamp}_est.wav"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    t0 = time.perf_counter()
    wav_path, used_prompt = generate_long_from_emotion(
        emotion=args.emotion,
        total_sec=int(args.total),
        chunk_sec=int(args.chunk),
        overlap_sec=float(args.overlap),
        outfile=out,
        override_prompt=args.prompt,
        seed=int(args.seed),
        variant=int(args.variant),
        mode=(None if not args.mode or args.mode.lower()=="none" else args.mode),
        model=model,
        meter_cb=meter_cb,
    )
    t1 = time.perf_counter()

    wall_sec   = t1 - t0
    active_sec = active_so_far + float(args.overhead_sec)

    print("\n=== Timing ===")
    print(f"Wall time:   {wall_sec:.2f}s")
    print(f"Active time: {active_sec:.2f}s  (sum of model.generate + {args.overhead_sec:.1f}s overhead)")

    print("\n=== Cost estimate (edit RATES_PER_MIN as needed) ===")
    cost_table = {}
    for name, rate in RATES_PER_MIN.items():
        cost = rate * (active_sec / 60.0)
        cost_table[name] = round(cost, 4)
        print(f"{name:18s}  ≈  ${cost:,.3f} per track")

    # write a small JSON report next to the file
    report = {
        "ts": stamp,
        "outfile": wav_path,
        "emotion": args.emotion,
        "prompt": used_prompt,
        "total_sec": args.total,
        "chunk_sec": args.chunk,
        "overlap_sec": args.overlap,
        "seed": args.seed,
        "variant": args.variant,
        "mode": args.mode,
        "per_chunk_sec": [round(x, 3) for x in per_chunk],
        "active_sec": round(active_sec, 3),
        "wall_sec": round(wall_sec, 3),
        "overhead_sec": float(args.overhead_sec),
        "cost_usd": cost_table,
    }
    repath = Path(args.logdir) / (Path(wav_path).stem + ".json")
    repath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report → {repath}")

if __name__ == "__main__":
    main()
