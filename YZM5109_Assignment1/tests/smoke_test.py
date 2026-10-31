"""End-to-end smoke test with a tiny random Qwen2 model (no network, CPU only).

It does not measure anything meaningful; it proves the pipeline runs: chat templating,
label masking, batched generation, PAL execution, LoRA training, resume, and statistics.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import a1lib as L

QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

def unit_tests():
    assert L.extract_answer("so 3+4=7. The answer is: 1,234.") == "1234"
    assert L.extract_answer("**The answer is: $18**") == "18"
    assert L.extract_answer("\\boxed{42}") == "42"
    assert L.extract_answer("we get 5 then 6.50") == "6.5"
    assert L.extract_answer("no digits") is None
    assert L.is_correct("18.0", "18") and not L.is_correct(None, "18")
    row = {"question": "Q?", "answer": "She has 2+3=<<2+3=5>>5 apples.\n#### 5"}
    p = L.parse_gsm8k(row)
    assert p["gold"] == "5" and "<<" not in p["rationale"], p
    assert L.run_pal("```python\ndef solution():\n    return 3*4\n```") == ("12", "ok")
    assert L.run_pal("```python\ndef solution():\n    while True: pass\n```", timeout=1)[1] == "timeout"
    assert L.run_pal("```python\ndef solution():\n    return 1/0\n```")[1].startswith("error")
    assert L.run_pal("def solution():\n    return 2.5")[0] == "2.5"  # bare / unterminated code
    t = L.mcnemar_exact([True, False, False, True], [True, True, True, True])
    assert t["b01"] == 2 and t["b10"] == 0 and abs(t["p_value"] - 0.5) < 1e-9, t
    lo, hi = L.bootstrap_ci([True] * 50 + [False] * 50, n_boot=2000)
    assert 0.38 < lo < 0.5 < hi < 0.62, (lo, hi)
    print("unit tests ok")


def tiny_model_and_tokenizer():
    corpus = ["The answer is: 12", "Solve the following math problem.", "def solution(): return 3",
              "Question: Tom has 3 apples and buys 4 more. How many?", "Let's think step by step"] * 50
    raw = Tokenizer(models.BPE(unk_token=None))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    raw.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = QWEN_TEMPLATE
    tok.padding_side = "left"
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg), tok


def main():
    unit_tests()
    model, tok = tiny_model_and_tokenizer()
    rows = [{"question": f"Tom has {i} apples and buys 4 more. How many?",
             "answer": f"He has {i}+4=<<{i}+4={i+4}>>{i+4}.\n#### {i+4}"} for i in range(12)]
    items = [L.parse_gsm8k(r) for r in rows]

    # Label masking: the only supervised tokens must be the assistant completion.
    ex = L.tokenize_sft(tok, items[0], max_len=512)
    sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    decoded = tok.decode(sup)
    assert decoded.startswith(items[0]["rationale"][:5]) and decoded.endswith("<|im_end|>\n"), decoded

    with tempfile.TemporaryDirectory() as out:
        res = {}
        for name, build, scorer in [("C1_direct", L.build_direct, L.score_text),
                                    ("C2_cot", L.build_cot, L.score_text),
                                    ("C3_pal", L.build_pal, L.run_pal),
                                    ("Cx_fewshot", L.make_fewshot_builder(items[:2]), L.score_text)]:
            res[name] = L.run_condition(name, model, tok, items, build, scorer, out,
                                        batch_size=5, max_new_tokens=8)
            assert len(res[name]) == len(items)
        # Resume path: a second call must load from disk without regenerating.
        again = L.run_condition("C1_direct", model, tok, items, L.build_direct, L.score_text, out)
        assert [r["output"] for r in again] == [r["output"] for r in res["C1_direct"]]

        cfg = L.LoraTrainConfig(r=4, alpha=8, max_steps=6, micro_batch_size=2, grad_accum=2, log_every=2,
                                max_len=128)
        model, stats = L.train_lora(model, tok, items, cfg, out_dir=out)
        assert stats["trainable_params"] > 0 and stats["trainable_pct"] < 100
        assert os.path.exists(os.path.join(out, "lora_adapter", "adapter_config.json"))
        losses = [x["loss"] for x in stats["loss_log"]]
        print("loss curve:", [round(x, 3) for x in losses])
        assert losses[-1] < losses[0], "LoRA training did not reduce loss on a trivially learnable task"

        res["C4_lora"] = L.run_condition("C4_lora", model, tok, items, L.build_direct, L.score_text, out,
                                         batch_size=4, max_new_tokens=8)
        summary = L.summarize(res, n_boot=200)
        tests = L.pairwise_tests(res)
        print(json.dumps(summary, indent=1)[:600])
        print(L.latex_results_table(summary)[:300])
        assert len(tests) == 10
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
