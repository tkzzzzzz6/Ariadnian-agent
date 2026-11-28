"""仅在固定 Base 开发校准通过后封存确认输入。"""
import argparse
import json
from .next_stage import seal_confirmation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'final-sources', 'plan', 'run', 'model-assets', 'output'):
        p.add_argument('--' + name, required=True)
    print(json.dumps(seal_confirmation(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
