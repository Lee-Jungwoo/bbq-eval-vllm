#!/usr/bin/env python3
"""Pick the label-token variant ("A" vs " A") that encodes to a single token.

" A" and "A" are different tokens in most tokenizers. For argmax scoring on the
assistant's first token we need each option letter to be exactly one token, so we
encode both variants (without special tokens) and pick the one where all three
letters are single tokens. When both work, the plain variant is preferred because
chat templates place the assistant content immediately after the generation
prompt, with no leading space.
"""
import argparse
import json

LETTERS = ["A", "B", "C"]


def choose_label_tokens(tokenizer):
    """Return {"variant", "option_letters", "label_token_ids", "candidates"}.

    label_token_ids maps the plain letter ("A"/"B"/"C") to the chosen token id.
    Raises ValueError if neither variant is single-token for all three letters.
    """
    candidates = {}
    for variant, prefix in [("plain", ""), ("space", " ")]:
        strings = [prefix + letter for letter in LETTERS]
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in strings]
        candidates[variant] = {
            "option_letters": strings,
            "ids": ids,
            "all_single": all(len(i) == 1 for i in ids),
        }

    for variant in ["plain", "space"]:
        cand = candidates[variant]
        if cand["all_single"]:
            return {
                "variant": variant,
                "option_letters": cand["option_letters"],
                "label_token_ids": {
                    letter: cand["ids"][i][0] for i, letter in enumerate(LETTERS)
                },
                "candidates": {
                    v: {"option_letters": c["option_letters"], "ids": c["ids"]}
                    for v, c in candidates.items()
                },
            }

    raise ValueError(
        f"Neither 'A' nor ' A' encodes to a single token for all letters: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--revision", default=None, help="HF revision (branch/tag/sha)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    result = choose_label_tokens(tokenizer)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
