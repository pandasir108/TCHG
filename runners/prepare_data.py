from __future__ import annotations

import argparse

from tchg.runner import prepare
from tchg.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/epinions_70_15_15.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = prepare(load_config(args.config), force=args.force)
    print(path)


if __name__ == "__main__":
    main()
