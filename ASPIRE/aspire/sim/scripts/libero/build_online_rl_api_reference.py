#!/usr/bin/env python3
"""Generate reviewable V1 API documentation without starting perception services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspire.sim.cap.integrations.franka.libero_reduced_skill_library import (
    FrankaLiberoApiReducedSkillLibrary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Documentation only: bypass service-heavy __init__. functions() returns
    # bound methods and does not touch the simulator. Molmo is explicitly absent.
    api = FrankaLiberoApiReducedSkillLibrary.__new__(
        FrankaLiberoApiReducedSkillLibrary
    )
    api._molmo_available = False
    functions = api.functions()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "franka_libero_reduced.md"
    temporary = reference.with_suffix(".md.tmp")
    temporary.write_text(
        "# Franka LIBERO reduced API (Online-RL V1)\n\n"
        "Molmo is disabled and is not part of this contract.\n\n"
        + api.combined_doc()
        + "\n"
    )
    temporary.replace(reference)
    print(
        json.dumps(
            {
                "output": str(reference),
                "functions": sorted(functions),
                "function_count": len(functions),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
