"""Builds YZM5109_A1_OptionC.ipynb, embedding src/a1lib.py verbatim so the notebook is self-contained."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = open(os.path.join(HERE, "src", "a1lib.py")).read()

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# YZM5109 – Assignment I · Option C: LoRA Fine-Tuning vs. Prompt Engineering

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (decoder-only) · **Task / data:** grade-school math word problems, `openai/gsm8k` (exact-match on the final number)

| Condition | Model | Prompt |
|---|---|---|
| **C1** | original | zero-shot direct |
| **C2** | original | zero-shot chain-of-thought |
| **C3** | original | Program-Aided LM (model writes Python, we execute it) |
| **C4** | LoRA fine-tuned (r=16, α=32) | *exactly* the C1 direct prompt |
| C5 (optional) | original | 4-shot with worked solutions |

All conditions share the same test items and the same greedy decoding (`do_sample=False`, `max_new_tokens=512`).

**How to run:** `Runtime → Change runtime type → T4 GPU` (or better), then `Runtime → Run all`.
Run once with `QUICK_RUN = True` (≈5 min) to check everything works, then set it to `False` for the real run.
Predictions are written to Google Drive after every batch, so if Colab disconnects just run all cells again; finished work is skipped.
""")

code(r"""
# Tested versions (same versions the pipeline was smoke-tested with).
!pip -q install "transformers==5.22.1" "peft==0.21.2" "datasets==5.1.0" accelerate
""")

code("%%writefile a1lib.py\n" + LIB)

md("## 1 · Configuration")
code(r"""
QUICK_RUN   = True          # True: 20 test items + 10 training steps, to verify the pipeline. Set False for the real run.
USE_DRIVE   = True          # store results on Google Drive so a disconnect does not lose work
RUN_FEWSHOT = False         # optional 5th condition (4-shot); adds ~1 condition worth of runtime

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
N_TEST      = None          # None = full GSM8K test set (1319). Use e.g. 500 if you are short on GPU time.
SEED        = 42
GEN_BATCH   = 32            # lowered automatically on OOM
MAX_NEW_TOK = 512           # identical for every condition

import a1lib as L
LORA_CFG = L.LoraTrainConfig(
    r=16, alpha=32, dropout=0.05,
    lr=2e-4, micro_batch_size=4, grad_accum=4,   # effective batch 16
    max_steps=300, max_len=512, seed=SEED,
)

if QUICK_RUN:
    N_TEST, LORA_CFG.max_steps, LORA_CFG.log_every = 20, 10, 2

RUN_NAME = f"{MODEL_ID.split('/')[-1]}_n{N_TEST or 'all'}_steps{LORA_CFG.max_steps}" + ("_quick" if QUICK_RUN else "")
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = f"/content/drive/MyDrive/YZM5109_A1/{RUN_NAME}"
else:
    OUT_DIR = f"/content/YZM5109_A1/{RUN_NAME}"
import os; os.makedirs(OUT_DIR, exist_ok=True)
print("Results ->", OUT_DIR)
""")

md("## 2 · Data")
code(r"""
import random, json, torch
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")
train_all = [L.parse_gsm8k(r) for r in ds["train"]]
test_all  = [L.parse_gsm8k(r) for r in ds["test"]]

rng = random.Random(SEED)
test_items = test_all if N_TEST is None else rng.sample(test_all, N_TEST)

# Few-shot exemplars come from the training split, never from the test split.
fewshot_exemplars = random.Random(SEED + 1).sample(train_all, 4)
print(f"train={len(train_all)}  test(eval)={len(test_items)}")
print(test_items[0])
""")

md("## 3 · Original model: C1–C3 (prompting only)")
code(r"""
from transformers import AutoModelForCausalLM, AutoTokenizer

major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
DTYPE = torch.bfloat16 if major >= 8 else torch.float16     # T4 (sm75) -> fp16, L4/A100 -> bf16
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="cuda")
model.eval()

ENV = {"gpu": torch.cuda.get_device_name(0), "dtype": str(DTYPE), "torch": torch.__version__,
       "model": MODEL_ID, "n_test": len(test_items), "max_new_tokens": MAX_NEW_TOK, "seed": SEED}
print(ENV)
print(L.count_params(model))
""")

code(r"""
results = {}
results["C1_direct"] = L.run_condition("C1_direct", model, tok, test_items, L.build_direct, L.score_text,
                                       OUT_DIR, batch_size=GEN_BATCH, max_new_tokens=MAX_NEW_TOK)
""")
code(r"""
results["C2_cot"] = L.run_condition("C2_cot", model, tok, test_items, L.build_cot, L.score_text,
                                    OUT_DIR, batch_size=GEN_BATCH, max_new_tokens=MAX_NEW_TOK)
""")
code(r"""
results["C3_pal"] = L.run_condition("C3_pal", model, tok, test_items, L.build_pal, L.run_pal,
                                    OUT_DIR, batch_size=GEN_BATCH, max_new_tokens=MAX_NEW_TOK)
""")
code(r"""
if RUN_FEWSHOT:
    results["C5_fewshot"] = L.run_condition("C5_fewshot", model, tok, test_items,
                                            L.make_fewshot_builder(fewshot_exemplars), L.score_text,
                                            OUT_DIR, batch_size=max(1, GEN_BATCH // 2), max_new_tokens=MAX_NEW_TOK)
""")

md(r"""
## 4 · C4: LoRA fine-tuning, evaluated with the C1 direct prompt
Training input = the C1 direct prompt; target = GSM8K reference solution (calculator tags removed) + `The answer is: N`.
Loss is computed only on the answer tokens. If an adapter already exists in `OUT_DIR` (e.g. after a disconnect) it is reloaded instead of retrained.
""")
code(r"""
from peft import PeftModel
adapter_dir = os.path.join(OUT_DIR, "lora_adapter")
stats_path  = os.path.join(OUT_DIR, "train_stats.json")
if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")) and os.path.exists(stats_path):
    print("Reloading existing adapter from", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    train_stats = json.load(open(stats_path))
else:
    model, train_stats = L.train_lora(model, tok, train_all, LORA_CFG, out_dir=OUT_DIR)
print({k: v for k, v in train_stats.items() if k != "loss_log"})
""")
code(r"""
results["C4_lora"] = L.run_condition("C4_lora", model, tok, test_items, L.build_direct, L.score_text,
                                     OUT_DIR, batch_size=GEN_BATCH, max_new_tokens=MAX_NEW_TOK)
""")

md("## 5 · Analysis")
code(r"""
import pandas as pd
LABELS = {"C1_direct": "C1 Zero-shot direct", "C2_cot": "C2 Zero-shot CoT", "C3_pal": "C3 PAL",
          "C4_lora": "C4 LoRA FT + direct", "C5_fewshot": "C5 4-shot"}
results = {k: results[k] for k in sorted(results)}   # C1, C2, C3, C4, (C5)
summary = L.summarize(results)
tests   = L.pairwise_tests(results)
df  = pd.DataFrame(summary).assign(label=lambda d: d.condition.map(LABELS))
dft = pd.DataFrame(tests)
display(df.round(4)); display(dft.round(4))

pal_status = pd.Series([r["status"] for r in results["C3_pal"]]).value_counts()
print("PAL execution status:\n", pal_status)
""")
code(r"""
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9})

fig, ax = plt.subplots(figsize=(3.5, 2.4))
acc = 100 * df.accuracy
err = [100 * (df.accuracy - df.ci_low), 100 * (df.ci_high - df.accuracy)]
ax.bar(df.label.str.split(" ").str[0], acc, yerr=err, capsize=3, color="#4C72B0")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.set_title("GSM8K accuracy (95% CI)")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_accuracy.pdf"); fig.savefig(f"{OUT_DIR}/fig_accuracy.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.scatter(df.mean_gen_tokens, acc, color="#C44E52")
for _, r in df.iterrows():
    ax.annotate(r.label.split(" ")[0], (r.mean_gen_tokens, 100 * r.accuracy), textcoords="offset points", xytext=(4, 2))
ax.set_xlabel("Mean generated tokens"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy vs. inference cost")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_cost.pdf"); fig.savefig(f"{OUT_DIR}/fig_cost.png", dpi=200); plt.show()

if train_stats.get("loss_log"):
    lg = pd.DataFrame(train_stats["loss_log"])
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(lg.step, lg.loss, alpha=.35, color="#55A868")
    ax.plot(lg.step, lg.loss.rolling(10, min_periods=1).mean(), color="#55A868")
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Train loss"); ax.set_title("LoRA training loss")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_loss.pdf"); fig.savefig(f"{OUT_DIR}/fig_loss.png", dpi=200); plt.show()
""")
code(r"""
# Qualitative examples for the error-analysis section of the report.
by_idx = {k: {r["idx"]: r for r in v} for k, v in results.items()}
def pick(cond_right, cond_wrong, k=3):
    return [(by_idx[cond_right][i], by_idx[cond_wrong][i]) for i in by_idx[cond_right]
            if by_idx[cond_right][i]["correct"] and not by_idx[cond_wrong][i]["correct"]][:k]
examples = {
    "C4_right_C1_wrong": pick("C4_lora", "C1_direct"),
    "C1_right_C4_wrong": pick("C1_direct", "C4_lora"),
    "C3_right_C2_wrong": pick("C3_pal", "C2_cot"),
    "C2_right_C3_wrong": pick("C2_cot", "C3_pal"),
}
for k, v in examples.items():
    print("=" * 20, k, len(v))
    for a, b in v[:1]:
        print("Q:", a["question"][:300], "\nGOLD:", a["gold"])
        print("--- right:", a["output"][:500]); print("--- wrong:", b["output"][:500])
""")
code(r"""
final = {"env": ENV, "summary": summary, "pairwise_mcnemar": tests,
         "train": {k: v for k, v in train_stats.items() if k != "loss_log"},
         "pal_status": pal_status.to_dict(), "examples": examples}
json.dump(final, open(f"{OUT_DIR}/results.json", "w"), indent=2)
open(f"{OUT_DIR}/results_table.tex", "w").write(L.latex_results_table(summary, LABELS))
df.to_csv(f"{OUT_DIR}/summary.csv", index=False); dft.to_csv(f"{OUT_DIR}/mcnemar.csv", index=False)

# Small zip for the report: everything except the LoRA weights (those stay on Drive / can go to the HF Hub).
import zipfile
zip_path = "/content/YZM5109_A1_results_small.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fnames in os.walk(OUT_DIR):
        if "lora_adapter" in root:
            continue
        for fn in fnames:
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, OUT_DIR))
print("Saved:", zip_path, f"{os.path.getsize(zip_path) / 1e6:.1f} MB")
try:
    from google.colab import files; files.download(zip_path)
except Exception as e:
    print("Download manually from the Files panel:", e)
""")

md("## 5b · Compact results to copy-paste (if you cannot upload files)")
code(r"""
compact = {"env": ENV, "summary": summary, "pairwise_mcnemar": tests,
           "train": {k: v for k, v in train_stats.items() if k != "loss_log"},
           "pal_status": pal_status.to_dict(),
           "loss_every_10": [round(x["loss"], 4) for x in train_stats.get("loss_log", [])][::10],
           "examples": {k: [{"q": a["question"][:400], "gold": a["gold"], "right": a["output"][:600],
                             "wrong": b["output"][:600]} for a, b in v[:2]] for k, v in examples.items()}}
print(json.dumps(compact, default=float))
""")

md(r"""
## 6 · (Optional) Push the LoRA adapter to the Hugging Face Hub
Add an `HF_TOKEN` secret in Colab (key icon on the left) with *write* access, then run.
""")
code(r"""
PUSH_TO_HUB = False
HUB_REPO    = "your-username/qwen2.5-1.5b-gsm8k-lora-r16"
if PUSH_TO_HUB:
    from google.colab import userdata
    from huggingface_hub import login
    login(userdata.get("HF_TOKEN"))
    model.push_to_hub(HUB_REPO, private=True)
    tok.push_to_hub(HUB_REPO, private=True)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [ln for ln in src.splitlines(keepends=True)]
out = os.path.join(HERE, "YZM5109_A1_OptionC.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
