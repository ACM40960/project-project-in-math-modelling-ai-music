# part2_musicgen/longgen.py
import os, random, numpy as np, torch, contextlib, time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from .prompting import build_prompt_and_params  # params_dict not needed here


def _seed_everything(seed: int | None):
    # Make results repeatable when a seed is given
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_long_from_emotion(
    emotion: str,
    total_sec: int = 90,
    chunk_sec: int | None = None,
    overlap_sec: float = 1.0,
    outfile: str | None = None,
    override_prompt: str | None = None,
    seed: int | None = None,
    variant: int | None = None,
    mode: str | None = None,
    model=None,                     # allow cached model injection (Streamlit)
    progress_cb=None, 
    meter_cb=None,    
):
    """
    meter_cb is called after each model.generate(...) with:
        (i_done, n_total, chunk_sec, active_sec_so_far, eta_sec)
    where:
        - chunk_sec = time spent inside model.generate for that chunk
        - active_sec_so_far = sum of all chunk_sec so far
        - eta_sec = (active_sec_so_far / i_done) * (n_total - i_done)
    """
    _seed_everything(seed)

    # Create a model if caller didn't pass one
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)

    # Build the text prompt + generation params for this emotion
    prompt, params = build_prompt_and_params(
        emotion, override_prompt=override_prompt, seed=seed, variant=variant, mode=mode
    )
    use_chunk = chunk_sec if chunk_sec is not None else params.duration

    # How many chunks do we need to hit total_sec?
    total_chunks = max(1, int(np.ceil(total_sec / use_chunk)))

    # Set generation hyperparameters
    model.set_generation_params(
        duration=int(use_chunk),
        use_sampling=params.use_sampling,
        top_k=params.top_k,
        top_p=params.top_p,
        temperature=params.temperature,
        cfg_coef=params.cfg_coef,
    )

    # Use AMP on CUDA for speed; no-op on CPU
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    # Generate chunks (timed per chunk)
    pieces = []
    active_total = 0.0  # sum of model.generate seconds

    # First chunk (timed)
    t0 = time.perf_counter()
    with autocast_ctx:
        wav = model.generate(descriptions=[prompt])[0].cpu()  # (C, T) or (T,)
    t1 = time.perf_counter()
    dt = t1 - t0
    active_total += dt
    if meter_cb:
        eta = (active_total / 1.0) * (total_chunks - 1)
        meter_cb(1, total_chunks, dt, active_total, eta)
    if progress_cb:
        progress_cb(1, total_chunks)
    pieces.append(wav)

    # Remaining chunks with optional crossfade
    sr = 32000
    n_over = int(max(0.0, overlap_sec) * sr)

    def _slast(x, start: int | None, end: int | None):
        """Slice along the last dimension (time)."""
        idx = [slice(None)] * x.dim()
        idx[-1] = slice(start, end)
        return x[tuple(idx)]

    for i in range(2, total_chunks + 1):
        t0 = time.perf_counter()
        with autocast_ctx:
            w = model.generate(descriptions=[prompt])[0].cpu()
        t1 = time.perf_counter()
        dt = t1 - t0
        active_total += dt

        # seam handling
        if n_over > 0:
            tail = _slast(pieces[-1], -n_over, None)
            head = _slast(w, 0, n_over)    
            fade = torch.linspace(0, 1, steps=n_over, device=w.device)
            if tail.dim() >= 2:   # (C, n_over)
                fade = fade.unsqueeze(0)  # (1, n_over)
            x = tail * (1 - fade) + head * fade
            kept = _slast(pieces[-1], 0, -n_over)
            pieces[-1] = torch.cat([kept, x], dim=-1)
            pieces.append(_slast(w, n_over, None)) 
        else:
            pieces.append(w)

        if meter_cb:
            avg = active_total / i
            eta = avg * (total_chunks - i)
            meter_cb(i, total_chunks, dt, active_total, eta)
        if progress_cb:
            progress_cb(i, total_chunks)

    # Slight headroom to avoid clipping
    full = torch.cat(pieces, dim=-1) * 0.90

    out = outfile or f"outputs/{emotion}_long.wav"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    audio_write(out[:-4], full, sr, strategy="loudness")

    return out, prompt
