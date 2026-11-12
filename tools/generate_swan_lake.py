#!/usr/bin/env python3
"""Regenerate the original full-body Swan Lake CSV and phrase map."""
import argparse
from pathlib import Path
import sys

# A checkout export must use this checkout's score, even when another release
# is installed or the script is invoked from an unrelated working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from jc04_dancing.choreography import write_swan_lake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=94.0)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'src/jc04_dancing/assets/motions')
    args = parser.parse_args()
    metadata = write_swan_lake(args.output_dir / 'swan_lake.csv', args.output_dir / 'swan_lake.phrases.json',
                               duration_s=args.duration, fps=args.fps)
    print(f"Generated {metadata['sample_count']} full-body samples over {metadata['duration_s']:g} s in {args.output_dir}")


if __name__ == '__main__':
    main()
