"""Post-hoc analysis of the full run, from the per-item JSONL predictions.

Usage: python analysis/deep_analysis.py results/full_run [gsm8k_test.jsonl]
Writes analysis.json, fig_steps.pdf/png and analysis_tables.tex into that folder.
"""
import ast
import json
import operator
import os
import re
import sys

C = ["C1_direct", "C2_cot", "C3_pal", "C4_lora"]
SHORT = {"C1_direct": "C1", "C2_cot": "C2", "C3_pal": "C3", "C4_lora": "C4"}

# ---- arithmetic checker: evaluates the full expression left of '=' ---------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_EQ = re.compile(r"([0-9.,\s+\-*/×x÷()]+?)\s*=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\d/])")


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError


def normalise_latex(t: str) -> str:
    t = re.sub(r"\\(?:times|cdot)", "*", t)
    t = re.sub(r"\\div", "/", t)
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)
    return t.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ").replace("$", " ")


def check_equations(text: str) -> tuple[int, int]:
    ok = bad = 0
    for lhs, rhs in _EQ.findall(normalise_latex(text)):
        expr = lhs.replace(",", "").replace("×", "*").replace("÷", "/")
        expr = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", expr).strip()
        if not re.search(r"\d\s*[+\-*/]\s*[\d(]", expr) or expr[0] in "*/+" or "x" in expr:
            continue  # need a real binary operation and no stray variables
        try:
            v = _eval(ast.parse(expr, mode="eval").body)
            c = float(rhs.replace(",", ""))
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError, RecursionError):
            continue
        if abs(v - c) <= max(0.01, 0.005 * abs(c)):
            ok += 1
        else:
            bad += 1
    return ok, bad


def main(d):
    R = {c: {r["idx"]: r for r in map(json.loads, open(os.path.join(d, f"{c}.jsonl")))} for c in C}
    idx = sorted(R["C1_direct"])
    n = len(idx)
    # Reasoning steps of the reference solution = non-empty lines of the GSM8K rationale.
    out = {"n": n}
    try:  # GSM8K test split, either a local jsonl (argv[2]) or the Hugging Face copy
        if len(sys.argv) > 2:
            test = [json.loads(l) for l in open(sys.argv[2])]
        else:
            from datasets import load_dataset
            test = load_dataset("openai/gsm8k", "main")["test"]
        q2steps = {r["question"].strip(): len([l for l in r["answer"].split("####")[0].splitlines() if l.strip()])
                   for r in test}
    except Exception:
        q2steps = None

    arith = {}
    for c in ["C1_direct", "C2_cot", "C4_lora"]:
        ok = bad = items_bad = wrong_bad = wrong = 0
        for i in idx:
            o, b = check_equations(R[c][i]["output"])
            ok += o; bad += b; items_bad += b > 0
            if not R[c][i]["correct"]:
                wrong += 1; wrong_bad += b > 0
        arith[c] = {"equations": ok + bad, "eq_per_item": (ok + bad) / n, "error_rate": bad / max(1, ok + bad),
                    "items_with_error": items_bad, "wrong_items": wrong, "wrong_with_arith_error": wrong_bad}
    out["arithmetic"] = arith

    cor = {c: [R[c][i]["correct"] for i in idx] for c in C}
    out["latex_pct"] = {c: 100 * sum(("\\(" in R[c][i]["output"]) or ("\\[" in R[c][i]["output"]) for i in idx) / n
                        for c in ["C1_direct", "C2_cot", "C4_lora"]}
    out["median_tokens"] = {c: sorted(R[c][i]["gen_tokens"] for i in idx)[n // 2] for c in C}
    out["oracle"] = {
        "C2|C3": sum(a or b for a, b in zip(cor["C2_cot"], cor["C3_pal"])) / n,
        "C1|C4": sum(a or b for a, b in zip(cor["C1_direct"], cor["C4_lora"])) / n,
        "any": sum(any(cor[c][k] for c in C) for k in range(n)) / n,
        "all_right": sum(all(cor[c][k] for c in C) for k in range(n)) / n,
        "all_wrong": sum(not any(cor[c][k] for c in C) for k in range(n)) / n,
    }
    agree = [k for k, i in enumerate(idx) if R["C2_cot"][i]["pred"] is not None
             and R["C2_cot"][i]["pred"] == R["C3_pal"][i]["pred"]]
    out["cot_pal_agreement"] = {"n": len(agree), "pct": 100 * len(agree) / n,
                                "acc_when_agree": sum(cor["C2_cot"][k] for k in agree) / len(agree),
                                "acc_when_disagree_cot": sum(cor["C2_cot"][k] for k in range(n) if k not in set(agree)) / (n - len(agree))}
    pal = [R["C3_pal"][i] for i in idx]
    executable = [r for r in pal if r["status"] == "ok" and r["pred"] is not None]
    out["pal"] = {"executable_pct": 100 * len(executable) / n,
                  "acc_executable": sum(r["correct"] for r in executable) / len(executable),
                  "no_output": sum(r["status"] == "no_output" for r in pal),
                  "runtime_error": sum(r["status"].startswith("error") for r in pal),
                  "timeout": sum(r["status"] == "timeout" for r in pal)}

    if q2steps:
        buckets = [(1, 2), (3, 3), (4, 4), (5, 5), (6, 99)]
        names = ["1-2", "3", "4", "5", "6+"]
        steps = [q2steps.get(R["C1_direct"][i]["question"]) for i in idx]
        table = {}
        for (lo, hi), nm in zip(buckets, names):
            ks = [k for k, s in enumerate(steps) if s is not None and lo <= s <= hi]
            table[nm] = {"n": len(ks), **{c: sum(cor[c][k] for k in ks) / max(1, len(ks)) for c in C}}
        out["acc_by_steps"] = table
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        for c, m in zip(C, "osD^"):
            ax.plot(names, [100 * table[nm][c] for nm in names], marker=m, label=SHORT[c])
        ax.set_xlabel("Reference solution steps"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs. problem length"); ax.legend(ncol=4, fontsize=7, loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"fig_steps.{ext}"), dpi=200)

    json.dump(out, open(os.path.join(d, "analysis.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
