"""Direct-script entrypoint: disable bytecode before importing immutable consumer."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd() / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("campaign", "assess"), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.mode == "campaign":
        from cst_research.experiments.covertype_campaign import main as execute
    else:
        from cst_research.experiments.covertype_assessment import main as execute
    execute()


if __name__ == "__main__":
    main()
