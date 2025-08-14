# part2_musicgen/prompting.py
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import random

@dataclass
class GenParams:
    duration: int          # seconds per chunk
    temperature: float
    top_k: int
    top_p: float
    cfg_coef: float
    two_step_cfg: bool = False
    use_sampling: bool = True

# Example prompts + generation params for each emotion
EMOTION_PRESETS: dict[str, List[Tuple[str, GenParams]]] = {
    "joy": [
        ("upbeat pop with bright synths and claps, energetic, major key",
         GenParams(8, 0.95, 240, 0.0, 2.1)),
        ("funky guitar and lively drums, upbeat groove, sunny major chords",
         GenParams(8, 1.00, 260, 0.0, 2.0)),
        ("bright synthwave arps and 4-on-the-floor kick, euphoric",
         GenParams(8, 0.98, 260, 0.0, 2.05)),
    ],
    "gratitude": [
        ("warm piano arpeggios and soft pads, hopeful ambient, gentle",
         GenParams(8, 0.90, 200, 0.0, 2.2)),
        ("soft strings and subtle piano, comforting and reflective, major key",
         GenParams(8, 0.85, 200, 0.0, 2.3)),
    ],
    "love": [
        ("romantic acoustic guitar with soft strings, intimate, major key",
         GenParams(8, 0.90, 200, 0.0, 2.2)),
        ("slow ballad piano with lush pads, tender and warm",
         GenParams(8, 0.88, 200, 0.0, 2.25)),
    ],
    "calm": [
        ("slow ambient pads, warm synth, gentle reverb, no drums",
         GenParams(8, 0.82, 170, 0.0, 2.2)),
        ("lo-fi chill keys, vinyl noise, sparse textures, very relaxed",
         GenParams(8, 0.80, 180, 0.0, 2.0)),
    ],
    "sadness": [
        ("slow piano with mellow strings, spacious reverb, minor key",
         GenParams(8, 0.85, 180, 0.0, 2.3)),
        ("lo-fi piano and soft pads, sparse and wistful, minor harmonies",
         GenParams(8, 0.88, 190, 0.0, 2.2)),
        ("soft felt piano with long tails, intimate room, melancholy",
         GenParams(8, 0.84, 180, 0.0, 2.25)),
    ],
    "fear": [
        ("dark drones and sparse percussion, tense cinematic ambience",
         GenParams(8, 0.85, 200, 0.0, 2.25)),
        ("eerie textures and low pulses, suspenseful, minimal rhythm",
         GenParams(8, 0.82, 190, 0.0, 2.3)),
    ],
    "anger": [
        ("heavy drums and distorted guitar, aggressive, high energy",
         GenParams(8, 1.05, 260, 0.0, 1.9)),
        ("industrial percussion and gritty synths, driving and intense",
         GenParams(8, 1.00, 250, 0.0, 2.0)),
        ("dark bass and sharp drums, edgy, fast tempo",
         GenParams(8, 1.05, 270, 0.0, 1.95)),
    ],
    "neutral": [
        ("neutral chill lo-fi beat with soft keys",
         GenParams(8, 0.90, 200, 0.0, 2.0)),
        ("simple ambient bed with soft keys and pad, unobtrusive",
         GenParams(8, 0.88, 190, 0.0, 2.05)),
    ],
}

def _pick_variant(emotion: str, seed: Optional[int]=None, variant: Optional[int]=None) -> Tuple[str, GenParams]:
    """Pick one preset: exact index, seeded random, or full random."""
    e = (emotion or "neutral").lower()
    options = EMOTION_PRESETS.get(e, EMOTION_PRESETS["neutral"])
    if variant is not None:
        idx = max(0, min(variant, len(options)-1))
        return options[idx]
    if seed is not None:
        rnd = random.Random(seed)
        return rnd.choice(options)
    return random.choice(options)

def _nudge(params: GenParams, mode: Optional[str]) -> GenParams:
    """Lightly tweak params for variety, adherence, or cleaner output."""
    if not mode: 
        return params
    p = GenParams(**asdict(params))  # shallow copy
    if mode == "variety":
        p.temperature = min(1.2, p.temperature + 0.10)
        p.top_k = min(400, p.top_k + 50)
    elif mode == "adhere":
        p.cfg_coef = min(2.3, p.cfg_coef + 0.2)
    elif mode == "clean":
        p.temperature = max(0.6, p.temperature - 0.10)
        p.top_k = max(50, p.top_k - 50)
    return p

def build_prompt_and_params(
    emotion: str,
    override_prompt: Optional[str]=None,
    seed: Optional[int]=None,
    variant: Optional[int]=None,
    mode: Optional[str]=None,
) -> Tuple[str, GenParams]:
    """Return (prompt, params) with optional mode tweaks."""
    prompt, params = _pick_variant(emotion, seed=seed, variant=variant)
    params = _nudge(params, mode)
    if override_prompt:
        prompt = override_prompt
    return prompt, params

def params_dict(params: GenParams) -> dict:
    """Dict form of GenParams (for logging/export)."""
    return asdict(params)
