"""Run in the independent experiments/ms_swift environment: python -m cpt --help."""
import argparse
import json
from pathlib import Path

from .assets import fetch_tokenizer
from .core import audit, prepare
from .launch import launch, plan
from .schema import default_recipe, METHODS


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B / MS-SWIFT CPT preparation; never trains by default")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--reviews", type=Path)
        p.add_argument("--allow-license", action="append", default=[])
        if name == "prepare":
            p.add_argument("--model-assets", required=True, type=Path)
            p.add_argument("--sequence-length", default=8192, type=int)
            p.add_argument("--seed", default=17, type=int)
            p.add_argument("--mode", choices=["preview", "training"], default="preview")
    p = sub.add_parser("fetch-tokenizer", help="download pinned inert tokenizer/config files, never weights")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("recipe", help="print an editable PT/SFT recipe")
    p.add_argument("--method", choices=sorted(METHODS), default="pt_lora")
    p.add_argument("--sequence-length", type=int)
    p.add_argument("--smoke", action="store_true", help="20 optimizer-step pilot, not an equal-token comparison")
    p = sub.add_parser("sft-prepare", help="source-bound single-turn SFT export; preview by default")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--source-corpus", required=True, type=Path)
    p.add_argument("--model-assets", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reviews", type=Path)
    p.add_argument("--source-reviews", type=Path)
    p.add_argument("--allow-license", action="append", default=[])
    p.add_argument("--sequence-length", default=4096, type=int)
    p.add_argument("--mode", choices=["preview", "training"], default="preview")
    p = sub.add_parser("plan")
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--strategy", choices=["ordered", "shuffled"], default="ordered")
    p.add_argument("--init-adapter", type=Path, help="source-bound CPT adapter for SFT continuation")
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--execute", action="store_true", help="opt in to real training after all guards pass")
    args = parser.parse_args()
    try:
        if args.command == "fetch-tokenizer":
            result = fetch_tokenizer(args.output)
        elif args.command == "recipe":
            result = default_recipe(args.method, sequence_length=args.sequence_length, smoke=args.smoke)
        elif args.command == "audit":
            result = audit(args.input, args.output, reviews_path=args.reviews, allowed_licenses=args.allow_license)
        elif args.command == "prepare":
            result = prepare(args.input, args.model_assets, args.output, reviews_path=args.reviews,
                             allowed_licenses=args.allow_license, mode=args.mode,
                             sequence_length=args.sequence_length, seed=args.seed)
        elif args.command == "sft-prepare":
            from .sft import prepare_sft
            result = prepare_sft(args.input, args.source_corpus, args.model_assets, args.output,
                                 reviews_path=args.reviews, source_reviews_path=args.source_reviews,
                                 allowed_licenses=args.allow_license, mode=args.mode,
                                 sequence_length=args.sequence_length)
        elif args.command == "plan":
            result = plan(args.prepared, args.recipe, args.output, strategy=args.strategy,
                          init_adapter=args.init_adapter)
        else:
            result = launch(args.plan, execute=args.execute)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"CPT: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
