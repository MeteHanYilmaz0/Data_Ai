"""Recompute every evaluation metric from the saved per-item predictions and export CSV/JSON files.

Usage: python evaluate.py results/full_run
Metrics (reported for every condition):
  1. Accuracy (exact match of the final number, tolerance 1e-4) with 95% bootstrap CI.
  2. Answer validity rate: % of outputs that follow the required answer format
     ("The answer is: <number>"); for PAL, % of programs that execute and return a number.
  3. Mean generated tokens (inference cost).
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import a1lib as L  # noqa: E402

CONDITIONS = ["C1_direct", "C2_cot", "C3_pal", "C4_lora"]
LABELS = {"C1_direct": "C1 Zero-shot direct", "C2_cot": "C2 Zero-shot CoT",
          "C3_pal": "C3 PAL (program-aided)", "C4_lora": "C4 LoRA FT + direct"}
_FORMAT = re.compile(r"answer is[:\s]*\**\s*\$?\s*-?\d", re.IGNORECASE)


def valid(rec):
    if rec["condition"] == "C3_pal":
        return rec["status"] == "ok" and rec["pred"] is not None
    return bool(_FORMAT.search(rec["output"]))


def main(run_dir):
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    results, rows = {}, []
    for c in CONDITIONS:
        recs = sorted((json.loads(l) for l in open(os.path.join(run_dir, f"{c}.jsonl"))), key=lambda r: r["idx"])
        results[c] = recs
        with open(os.path.join(pred_dir, f"{c}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "condition", "prediction", "reference", "correct", "valid_format",
                        "gen_tokens", "question", "model_output"])
            for r in recs:
                w.writerow([r["idx"], c, r["pred"], r["gold"], int(r["correct"]), int(valid(r)),
                            r["gen_tokens"], r["question"], r["output"]])
    summary = {r["condition"]: r for r in L.summarize(results)}
    for c in CONDITIONS:
        s = summary[c]
        rows.append({"condition": c, "label": LABELS[c], "n_test": s["n"],
                     "accuracy_pct": round(100 * s["accuracy"], 2),
                     "ci95_low_pct": round(100 * s["ci_low"], 2), "ci95_high_pct": round(100 * s["ci_high"], 2),
                     "valid_format_pct": round(100 * sum(map(valid, results[c])) / s["n"], 2),
                     "mean_gen_tokens": round(s["mean_gen_tokens"], 1)})
    tests = L.pairwise_tests(results)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"metrics": rows, "mcnemar": tests}, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    json.dump([{"sample_id": r["idx"], "condition": r["condition"], "prediction": r["pred"], "reference": r["gold"],
                "correct": r["correct"]} for c in CONDITIONS for r in results[c]],
              open(os.path.join(pred_dir, "predictions_all.json"), "w"))

    print("YZM5109 Assignment I - Option C - Qwen/Qwen2.5-1.5B-Instruct on openai/gsm8k (test split)")
    print("Metrics recomputed from saved predictions:", run_dir)
    print()
    hdr = f"{'Condition':<24}{'n':>6}{'Accuracy %':>12}{'95% CI':>17}{'Valid fmt %':>13}{'Mean tokens':>13}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci95_low_pct']:.1f}, {r['ci95_high_pct']:.1f}]"
        print(f"{r['label']:<24}{r['n_test']:>6}{r['accuracy_pct']:>12.2f}{ci:>17}{r['valid_format_pct']:>13.2f}"
              f"{r['mean_gen_tokens']:>13.1f}")
    print()
    print("Exact McNemar tests (b01 = only B correct, b10 = only A correct)")
    for t in tests:
        print(f"  {t['a']:<10} vs {t['b']:<10} dAcc={100 * t['delta_acc']:+6.2f} pts  b01={t['b01']:>3}  "
              f"b10={t['b10']:>3}  p={t['p_value']:.2e}")


if __name__ == "__main__":
    main(sys.argv[1])
