# part1_emotion/eval_report.py
import os, json, argparse, numpy as np, pandas as pd, torch, time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- paths (edit if needed) ----------
ROOT = Path(__file__).resolve().parents[1]  # C:\SonicAid_clean
CSV  = ROOT / "data" / "raw" / "goemotions_df.csv"
ROBERTA_DIR = ROOT / "saved_models" / "roberta-goemotion-final"
DISTIL_DIR  = ROOT / "saved_models" / "goemotions_distilbert_v3"  # optional
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_LENGTH = 48  # from your p99~37 tokens

def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def load_model(path: Path):
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(DEVICE).eval()
    try:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=False, local_files_only=True)
    # labels (prefer saved label_classes.json)
    lbl_json = path / "label_classes.json"
    if lbl_json.exists():
        classes = json.loads(lbl_json.read_text(encoding="utf-8"))
        id2label = [classes[i] for i in range(len(classes))]
    else:
        cfg = model.config
        id2label = [cfg.id2label.get(str(i), cfg.id2label.get(i, f"label_{i}")) for i in range(cfg.num_labels)]
    return model, tok, id2label

@torch.no_grad()
def logits_on(model, tok, texts):
    outs = []
    bs = 64
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        outs.append(model(**enc).logits.detach().cpu())
    return torch.cat(outs, dim=0).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=str, default=None, help="Write a compact metrics JSON here")
    ap.add_argument("--w-rob", type=float, default=0.4, help="Ensemble weight for RoBERTa")
    ap.add_argument("--w-dis", type=float, default=0.6, help="Ensemble weight for DistilBERT")
    args = ap.parse_args()

    # ----- data -----
    df = pd.read_csv(CSV).dropna(subset=["text","label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()
    # dedupe by text (majority)
    y_major = df.groupby("text")["label"].agg(lambda s: s.value_counts().idxmax())
    X = y_major.index.tolist()
    y = y_major.values.tolist()

    # ----- RoBERTa (canonical label space) -----
    rob_model, rob_tok, rob_labels = load_model(ROBERTA_DIR)
    rob_labels_norm = [_norm(x) for x in rob_labels]
    name2id_rob = {lab: i for i, lab in enumerate(rob_labels_norm)}

    # filter rows to labels present in RoBERTa
    y_norm = [_norm(v) for v in y]
    keep_idx = [i for i, lab in enumerate(y_norm) if lab in name2id_rob]
    if not keep_idx:
        raise RuntimeError("No rows match RoBERTa label set; check CSV vs model labels.")

    X = [X[i] for i in keep_idx]
    y_ids = np.array([name2id_rob[y_norm[i]] for i in keep_idx], dtype=int)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_ids, test_size=0.10, random_state=SEED, stratify=y_ids
    )
    print(f"Validation size: {len(X_val)} examples")

    # ----- logits: RoBERTa -----
    rob_logits = logits_on(rob_model, rob_tok, X_val)

    # ----- optional Distil + ensemble (aligned to RoBERTa) -----
    have_distil = DISTIL_DIR.exists()
    ens_logits = rob_logits.copy()
    if have_distil:
        dis_model, dis_tok, dis_labels = load_model(DISTIL_DIR)
        dis_labels_norm = [_norm(x) for x in dis_labels]
        distil_to_rob = {i: name2id_rob[l] for i, l in enumerate(dis_labels_norm) if l in name2id_rob}
        dis_logits = logits_on(dis_model, dis_tok, X_val)

        w_rob, w_dis = float(args.w_rob), float(args.w_dis)
        acc = w_rob * ens_logits
        for i_dis, i_rob in distil_to_rob.items():
            acc[:, i_rob] += w_dis * dis_logits[:, i_dis]
        ens_logits = acc
    else:
        print("⚠️ DistilBERT folder not found; evaluating RoBERTa-only (ensemble==RoBERTa).")

    # ----- helpers -----
    def stats(name, logits, labels, label_names):
        preds = logits.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average="macro")
        f1w = f1_score(labels, preds, average="weighted")
        print(f"\n=== {name} ===")
        print("acc:", acc)
        print("f1_macro:", f1m)
        print("f1_weighted:", f1w)
        print(classification_report(labels, preds, target_names=label_names, digits=3))
        return {"Accuracy": float(acc), "F1-macro": float(f1m), "F1-weighted": float(f1w)}

    # ----- print + collect -----
    metrics_out = {
        "dataset": str(CSV),
        "val_size": int(len(X_val)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {}
    }

    metrics_out["models"]["RoBERTa"] = stats("RoBERTa", rob_logits, y_val, rob_labels)

    name_ens = "Ensemble (RoBERTa+Distil)" if have_distil else "Ensemble=RoBERTa"
    metrics_out["models"][name_ens] = stats(name_ens, ens_logits, y_val, rob_labels)

    # ----- optional JSON file -----
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
        print(f"\n📁 Wrote metrics JSON → {outp}")

if __name__ == "__main__":
    main()
