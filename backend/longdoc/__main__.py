"""Run with ``python -m longdoc build --input docs.jsonl --output run``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import BuildConfig, build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("build", help="Build an audited corpus from structured JSONL")
    cmd.add_argument("--input", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--tokenizer", type=Path, help="Local Hugging Face tokenizer.json; no downloads")
    cmd.add_argument("--sequence-length", type=int, default=8192,
                     help="Limit in selected tokenizer units; default UTF-8 bytes are diagnostic, NOT model tokens")
    cmd.add_argument("--seed", type=int, default=17)
    cmd.add_argument("--near-threshold", type=float, default=0.92)
    cmd.add_argument("--near-action", choices=("candidate-only", "remove"), default="candidate-only",
                     help="Near matches share a split either way; default keeps both for human review")
    cmd.add_argument("--min-quality-score", type=float, default=0.0,
                     help="Optional heuristic filter in [0,1]; not human quality verification")
    cmd.add_argument("--purpose", choices=("assessment", "training"), default="assessment")
    cmd.add_argument("--allow-license", action="append", default=[],
                     help="Exact license identifier approved for this training use; repeat as needed")
    cmd.add_argument("--mix", action="append", default=[], metavar="DOMAIN=WEIGHT",
                     help="Target train-domain proportions by tokenizer units, without replacement")
    cmd.add_argument("--mix-token-budget", type=int,
                     help="Mix budget in selected units; whole documents only, no overshoot")
    audit = sub.add_parser("assess", help="Controlled dedup/quality diagnostics plus blank natural-data review templates")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--seed", type=int, default=17)
    audit.add_argument("--cases-per-type", type=int, default=20)
    audit.add_argument("--near-threshold", type=float, default=0.92)
    audit.add_argument("--near-action", choices=("candidate-only", "remove"), default="candidate-only")
    audit.add_argument("--tokenizer", type=Path)
    audit.add_argument("--sequence-length", type=int, default=8192)
    review = sub.add_parser("score-review", help="Score completed natural-data reviews; blank fields remain unreviewed")
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--kind", choices=("dedup", "quality"), required=True)
    args = parser.parse_args(argv)
    weights: dict[str, float] = {}
    try:
        if args.command == "assess":
            from .assessment import assess

            result = assess(args.input, args.output, seed=args.seed, cases_per_type=args.cases_per_type,
                            near_threshold=args.near_threshold, near_action=args.near_action, sequence_length=args.sequence_length,
                            tokenizer_path=str(args.tokenizer) if args.tokenizer else None)
            print(json.dumps({"output": str(args.output.resolve()), "controlled_dedup": result["controlled_dedup"],
                              "controlled_quality": result["controlled_quality"], "natural_review": result["natural_review"]},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "score-review":
            from .assessment import score_review

            result = score_review(args.review, args.output, args.kind)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        for item in args.mix:
            domain, value = item.rsplit("=", 1)
            if not domain or domain in weights:
                raise ValueError("--mix requires unique, nonempty domains")
            weights[domain] = float(value)
        config = BuildConfig(
            seed=args.seed, sequence_length=args.sequence_length,
            tokenizer_path=str(args.tokenizer) if args.tokenizer else None,
            near_threshold=args.near_threshold, near_action=args.near_action, min_quality_score=args.min_quality_score,
            purpose=args.purpose, allowed_licenses=tuple(args.allow_license),
            mix_weights=weights, mix_token_budget=args.mix_token_budget,
        )
        result = build(args.input, args.output, config)
    except (ValueError, OSError, ImportError) as exc:
        print(f"longdoc: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"],
                      "tokenizer": result["tokenizer"], "purpose": config.purpose}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
