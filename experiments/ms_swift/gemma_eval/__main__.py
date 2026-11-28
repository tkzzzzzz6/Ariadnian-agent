import argparse
import json
from pathlib import Path

from .core import prepare
from .runtime import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned Gemma evaluation; preparation does not load weights", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", allow_abbrev=False)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--model-assets", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--kind", choices=["benchmark", "profile", "nll", "chat"], default="benchmark")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p = sub.add_parser("run", allow_abbrev=False)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("verify-chat", allow_abbrev=False, help="Compare frozen inputs to pinned SWIFT without model weights")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("inspect-archive", allow_abbrev=False, help="Read-only relocated archive hashes; never grants execution")
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--original-root", required=True)
    p.add_argument("--adapter-relative", required=True)
    p.add_argument("--output", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(args.pop("input"), args.pop("model_assets"), args.pop("output"), **args)
        elif command == "verify-chat":
            from .chat import verify_plan
            from cpt.core import new_output, write_json
            result = verify_plan(args["plan"])
            if args["output"]:
                write_json(new_output(args["output"]) / "chat_template_verification.json", result)
        elif command == "inspect-archive":
            from .portable import inspect_archive
            from cpt.core import new_output, write_json
            result = inspect_archive(args["artifact_root"], args["original_root"], args["adapter_relative"])
            if args["output"]:
                write_json(new_output(args["output"]) / "archive_inspection.json", result)
        else:
            result = run(args.pop("plan"), **args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Gemma evaluation: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if command == "run" and result.get("executed") and result.get("failed_or_skipped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
