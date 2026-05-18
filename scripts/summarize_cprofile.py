#!/usr/bin/env python3
"""Write top cProfile functions by cumtime and tottime."""

from __future__ import annotations

import argparse
import io
import os
import pstats
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cProfile output.")
    parser.add_argument("--profile", required=True, help="Path to .cprof file")
    parser.add_argument("--out", required=True, help="Path to output txt file")
    parser.add_argument("--topn", type=int, default=50)
    args = parser.parse_args()

    profile_path = Path(args.profile).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not profile_path.exists():
        out_path.write_text(f"missing cprofile file: {profile_path}\n", encoding="utf-8")
        print(str(out_path))
        return

    buf = io.StringIO()
    buf.write(f"profile: {profile_path}\n")
    buf.write(f"pid: {os.getpid()}\n\n")

    stats = pstats.Stats(str(profile_path), stream=buf)

    buf.write("=== Top functions by cumulative time ===\n")
    stats.sort_stats("cumtime").print_stats(int(args.topn))

    buf.write("\n=== Top functions by total time ===\n")
    stats.sort_stats("tottime").print_stats(int(args.topn))

    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
