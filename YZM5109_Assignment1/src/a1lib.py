"""YZM5109 Assignment I, Option C: LoRA fine-tuning vs. prompt engineering on GSM8K.

Self-contained helper library. The Colab notebook embeds this exact file, and
tests/smoke_test.py exercises it end to end with a tiny randomly initialised model.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

# --------------------------------------------------------------------------------------
# Prompts. Every condition uses the same system prompt and the same answer-format line,
# so the only thing that changes between C1, C2 and C3 is the prompting technique.
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful assistant."
ANSWER_FORMAT = 'End your response with "The answer is: <number>".'

DIRECT_TEMPLATE = "Solve the following math problem.\n\nQuestion: {question}\n\n" + ANSWER_FORMAT

COT_TEMPLATE = (
    "Solve the following math problem. Let's think step by step: break the problem into "
    "smaller steps, write out every intermediate calculation, and check the result before "
    "answering.\n\nQuestion: {question}\n\n" + ANSWER_FORMAT
)

PAL_TEMPLATE = (
    "Write a Python function named `solution()` that solves the following math problem and "
    "returns the final numeric answer. Use descriptive variable names and simple arithmetic. "
    "Do not use input() or external libraries. Reply with a single ```python code block only."
    "\n\nQuestion: {question}"
)


def _msgs(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_direct(item: dict) -> list[dict]:
    return _msgs(DIRECT_TEMPLATE.format(question=item["question"]))


def build_cot(item: dict) -> list[dict]:
    return _msgs(COT_TEMPLATE.format(question=item["question"]))


def build_pal(item: dict) -> list[dict]:
    return _msgs(PAL_TEMPLATE.format(question=item["question"]))


def make_fewshot_builder(exemplars: list[dict]) -> Callable[[dict], list[dict]]:
    """Optional extra condition: k-shot with worked solutions as prior chat turns."""

    def build(item: dict) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in exemplars:
            msgs.append({"role": "user", "content": DIRECT_TEMPLATE.format(question=ex["question"])})
            msgs.append({"role": "assistant", "content": format_target(ex)})
        msgs.append({"role": "user", "content": DIRECT_TEMPLATE.format(question=item["question"])})
        return msgs

    return build


# --------------------------------------------------------------------------------------
# GSM8K parsing and answer extraction.
# --------------------------------------------------------------------------------------
_CALC_ANNOT = re.compile(r"<<[^>]*>>")
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def parse_gsm8k(row: dict) -> dict:
    rationale, gold = row["answer"].split("####")
    return {
        "question": row["question"].strip(),
        "rationale": _CALC_ANNOT.sub("", rationale).strip(),
        "gold": normalize_number(gold),
    }


def format_target(item: dict) -> str:
    """Fine-tuning target: GSM8K rationale (calculator tags stripped) plus the shared answer line."""
    return f"{item['rationale']}\nThe answer is: {item['gold']}"


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6g}"


def extract_answer(text: str) -> str | None:
    """Prefer the mandated 'The answer is: X' line, then \\boxed{X}, then the last number."""
    m = re.findall(r"answer is[:\s]*\**\s*\$?\s*(" + _NUM + ")", text, flags=re.IGNORECASE)
    if m:
        return normalize_number(m[-1])
    m = re.findall(r"\\boxed\{\s*\$?\s*(" + _NUM + ")", text)
    if m:
        return normalize_number(m[-1])
    m = re.findall(_NUM, text)
    return normalize_number(m[-1]) if m else None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# PAL: execute model-written code in an isolated subprocess with a timeout.
# --------------------------------------------------------------------------------------
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len)
    # Unterminated block (hit max_new_tokens) or bare code.
    return re.sub(r"^```(?:python|py)?\s*\n", "", text.strip())


def _print_last_expression(code: str) -> str:
    """Mimic a notebook/REPL: if the program ends with a bare expression (e.g. `total_cost`),
    print its value. Models frequently end PAL programs this way instead of calling print()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        is_print = (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "print")
        if not is_print:
            tree.body[-1] = ast.Expr(ast.Call(func=ast.Name("print", ast.Load()), args=[value], keywords=[]))
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return code


def run_pal(text: str, timeout: float = 5.0) -> tuple[str | None, str]:
    """Answer = return value of solution() if defined, else the last printed value or bare final expression."""
    code = _print_last_expression(extract_code(text))
    runner = (
        code
        + "\n\nif __name__ == '__main__':\n"
        + "    if callable(globals().get('solution')):\n"
        + "        print(solution())\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return None, ("error: " + err[-1]) if err else "error"
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None, "no_output"
    m = re.findall(_NUM, lines[-1])
    return (normalize_number(m[-1]) if m else None), "ok"


def score_text(text: str) -> tuple[str | None, str]:
    return extract_answer(text), "ok"


# --------------------------------------------------------------------------------------
# Batched greedy generation with OOM back-off.
# --------------------------------------------------------------------------------------
def make_generation_config(model, tok, max_new_tokens: int):
    from transformers import GenerationConfig

    eos = tok.eos_token_id
    eos_ids = sorted({i for i in [eos, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
                      if isinstance(i, int) and i >= 0 and i != tok.unk_token_id})
    return GenerationConfig(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids,
        repetition_penalty=1.0,
    )


@torch.no_grad()
def generate_batch(model, tok, msg_lists: list[list[dict]], gen_cfg) -> list[tuple[str, int]]:
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msg_lists]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, generation_config=gen_cfg)
    new = out[:, enc["input_ids"].shape[1]:]
    eos_ids = set(gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id])
    results = []
    for row in new.tolist():
        n = 0
        for t in row:
            n += 1
            if t in eos_ids:
                break
        results.append((tok.decode(row, skip_special_tokens=True).strip(), n))
    return results


def run_condition(
    name: str,
    model,
    tok,
    items: list[dict],
    build_msgs: Callable[[dict], list[dict]],
    scorer: Callable[[str], tuple[str | None, str]],
    out_dir: str,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    log_every: int = 1,
) -> list[dict]:
    """Evaluate one condition. Predictions stream to JSONL so a Colab disconnect can resume."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.jsonl")
    done: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[r["idx"]] = r
    todo = [i for i in range(len(items)) if i not in done]
    if not todo:
        print(f"[{name}] already complete ({len(done)} items), loading from {path}")
        return [done[i] for i in range(len(items))]

    gen_cfg = make_generation_config(model, tok, max_new_tokens)
    model.eval()
    # Sort by prompt length so batches pad less; results are re-ordered by idx at the end.
    todo.sort(key=lambda i: -len(build_msgs(items[i])[-1]["content"]))
    bs, pos, t_start, n_batches = batch_size, 0, time.time(), 0
    with open(path, "a") as f:
        while pos < len(todo):
            chunk = todo[pos:pos + bs]
            try:
                t0 = time.time()
                outs = generate_batch(model, tok, [build_msgs(items[i]) for i in chunk], gen_cfg)
                dt = time.time() - t0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[{name}] OOM, lowering batch size to {bs}")
                continue
            for i, (text, ntok) in zip(chunk, outs):
                pred, status = scorer(text)
                rec = {
                    "idx": i,
                    "condition": name,
                    "question": items[i]["question"],
                    "gold": items[i]["gold"],
                    "output": text,
                    "pred": pred,
                    "status": status,
                    "correct": is_correct(pred, items[i]["gold"]),
                    "gen_tokens": ntok,
                    "hit_max_tokens": ntok >= max_new_tokens,
                    "latency_s": dt / len(chunk),
                }
                done[i] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
            pos += len(chunk)
            n_batches += 1
            if n_batches % log_every == 0:
                acc = sum(done[j]["correct"] for j in done) / len(done)
                print(f"[{name}] {len(done)}/{len(items)}  running acc={acc:.3f}  "
                      f"elapsed={time.time() - t_start:.0f}s  bs={bs}")
    return [done[i] for i in range(len(items))]


# --------------------------------------------------------------------------------------
# LoRA fine-tuning (plain PyTorch loop: fewer moving parts than Trainer across versions).
# --------------------------------------------------------------------------------------
@dataclass
class LoraTrainConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lr: float = 2e-4
    micro_batch_size: int = 4
    grad_accum: int = 4
    max_steps: int = 300
    warmup_ratio: float = 0.05
    max_len: int = 512
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = True
    log_every: int = 10


def tokenize_sft(tok, item: dict, max_len: int) -> dict | None:
    """Tokenise one (direct prompt -> rationale) pair; loss only on the assistant tokens."""
    prompt_msgs = build_direct(item)
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": format_target(item)}], tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template does not extend the generation prompt; cannot mask labels safely.")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def _collate(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for k, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[k, :n] = torch.tensor(b["input_ids"])
        labels[k, :n] = torch.tensor(b["labels"])
        mask[k, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_lora(model, tok, train_items: list[dict], cfg: LoraTrainConfig, out_dir: str | None = None):
    from peft import LoraConfig, get_peft_model

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    lcfg = LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                      target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    model = get_peft_model(model, lcfg)
    trainable, total = count_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    data = [x for x in (tokenize_sft(tok, it, cfg.max_len) for it in train_items) if x is not None]
    device = next(model.parameters()).device
    base_dtype = next(p for p in model.parameters() if not p.requires_grad).dtype
    use_amp = device.type == "cuda" and base_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and base_dtype == torch.float16)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, cfg.max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    order: list[int] = []
    rng = random.Random(cfg.seed)

    def next_micro():
        nonlocal order
        if len(order) < cfg.micro_batch_size:
            fresh = list(range(len(data)))
            rng.shuffle(fresh)
            order += fresh
        idx, order = order[:cfg.micro_batch_size], order[cfg.micro_batch_size:]
        return _collate([data[i] for i in idx], tok.pad_token_id)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    model.train()
    log, t0, seen_tokens = [], time.time(), 0
    for step in range(cfg.max_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            batch = {k: v.to(device) for k, v in next_micro().items()}
            seen_tokens += int(batch["attention_mask"].sum())
            with torch.autocast(device_type=device.type, dtype=base_dtype, enabled=use_amp):
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        sched.step()
        log.append({"step": step + 1, "loss": step_loss, "lr": sched.get_last_lr()[0]})
        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(f"step {step + 1}/{cfg.max_steps}  loss={step_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.0f}s")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    model.eval()
    if cfg.gradient_checkpointing:
        model.base_model.model.gradient_checkpointing_disable()
    model.config.use_cache = True
    stats = {
        "lora": asdict(cfg),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "train_time_s": train_time,
        "train_examples_available": len(data),
        "examples_seen": cfg.max_steps * cfg.micro_batch_size * cfg.grad_accum,
        "tokens_seen": seen_tokens,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "loss_log": log,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(os.path.join(out_dir, "lora_adapter"))
        with open(os.path.join(out_dir, "train_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    return model, stats


# --------------------------------------------------------------------------------------
# Statistics: accuracy, bootstrap CI, exact McNemar test on paired predictions.
# --------------------------------------------------------------------------------------
def bootstrap_ci(correct: list[bool], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(correct)
    accs = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return accs[int(alpha / 2 * n_boot)], accs[int((1 - alpha / 2) * n_boot) - 1]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test. b01: a wrong & b right, b10: a right & b wrong."""
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * p)}


def summarize(results: dict[str, list[dict]], n_boot: int = 10000) -> list[dict]:
    rows = []
    for name, recs in results.items():
        corr = [r["correct"] for r in recs]
        lo, hi = bootstrap_ci(corr, n_boot=n_boot)
        n = len(recs)
        rows.append({
            "condition": name,
            "n": n,
            "accuracy": sum(corr) / n,
            "ci_low": lo,
            "ci_high": hi,
            "mean_gen_tokens": sum(r["gen_tokens"] for r in recs) / n,
            "truncated_pct": 100 * sum(r["hit_max_tokens"] for r in recs) / n,
            "no_answer_pct": 100 * sum(r["pred"] is None for r in recs) / n,
            "mean_latency_s": sum(r["latency_s"] for r in recs) / n,
        })
    return rows


def pairwise_tests(results: dict[str, list[dict]]) -> list[dict]:
    names = list(results)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [r["correct"] for r in results[names[i]]]
            b = [r["correct"] for r in results[names[j]]]
            t = mcnemar_exact(a, b)
            out.append({"a": names[i], "b": names[j], **t,
                        "delta_acc": (sum(b) - sum(a)) / len(a)})
    return out


def latex_results_table(rows: list[dict], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{GSM8K test accuracy (greedy decoding, identical settings). 95\% bootstrap CI.}",
        r"\label{tab:main}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Condition & Acc. (\%) & 95\% CI & Avg. tokens \\", r"\midrule",
    ]
    for r in rows:
        label = labels.get(r["condition"], r["condition"].replace("_", r"\_"))
        lines.append(f"{label} & {100 * r['accuracy']:.1f} & "
                     f"[{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}] & {r['mean_gen_tokens']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
