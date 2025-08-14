# tools/ab_test.py
import os, csv
from datetime import datetime
from part2_musicgen.longgen import generate_long_from_emotion
from part2_musicgen.prompting import build_prompt_and_params, params_dict

LOG = "experiments/log.tsv"
os.makedirs("outputs", exist_ok=True)      # where audio files go
os.makedirs("experiments", exist_ok=True)  # where the TSV log lives

def _write_tsv(path: str, row: dict):
    # Append a row to a tab-separated log; create header on first run.
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerow(row)

def run_pair(tag: str, kw: dict, seed: int | None = 42, mode: str | None = None, variant: int | None = None):
    """Generate one track and log the exact settings used."""
    emotion = kw["emotion"]
    # Lock prompt/params to the seed so the log matches the audio
    prompt, gparams = build_prompt_and_params(
        emotion, override_prompt=kw.get("override_prompt"), seed=seed, variant=variant, mode=mode
    )
    out = kw.get("outfile") or f"outputs/AB_{tag}.wav"
    path, used_prompt = generate_long_from_emotion(
        emotion=emotion,
        total_sec=kw.get("total_sec", 45),
        chunk_sec=kw.get("chunk_sec", 8),
        overlap_sec=kw.get("overlap_sec", 1.0),
        outfile=out,
        override_prompt=kw.get("override_prompt") or prompt,
        seed=seed,
        variant=variant,
        mode=mode,
    )

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "emotion": emotion,
        "seed": seed,
        "mode": mode or "default",
        "variant": variant if variant is not None else "",
        "total_sec": kw.get("total_sec", 45),
        "chunk_sec": kw.get("chunk_sec", 8),
        "overlap_sec": kw.get("overlap_sec", 1.0),
        "outfile": path,
        "prompt": used_prompt,
        **params_dict(gparams),  # duration, temperature, top_k, top_p, cfg_coef, ...
    }
    _write_tsv(LOG, row)
    print(f"{tag}: {path} | {used_prompt}")

if __name__ == "__main__":
    # Simple A/B: same emotion, tweak chunk/overlap
    pairs = [
        ("A", dict(emotion="sadness", total_sec=45, chunk_sec=8,  overlap_sec=1.0)),
        ("B", dict(emotion="sadness", total_sec=45, chunk_sec=10, overlap_sec=1.5)),
    ]
    for tag, kw in pairs:
        run_pair(tag, kw, seed=42, mode=None, variant=None)

    print(f"\nLogged to: {LOG}")
