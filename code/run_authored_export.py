"""Package complete two-sock geometry/features with a strict 1 GB cap."""
from pathlib import Path
import argparse
import json
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent
sys.path.insert(0, str(CODE))
sys.pycache_prefix = str(ROOT / "build-support/stocking-material/pycache")

from stocking_material.authored_export import plan_authored_export, write_authored_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=ROOT / "stocking/assets/authored-fit-v1")
    parser.add_argument("--features", type=Path, default=ROOT / "stocking/assets/authored-yarn-v1/budget-1gb")
    parser.add_argument("--output", type=Path, default=ROOT / "stocking/assets/authored-export-1gb-v1")
    parser.add_argument("--maximum-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--write", action="store_true", help="write only after all features are complete")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    extras = [(CODE / "AUTHORED_GEOMETRY_METHOD.md", "docs/AUTHORED_GEOMETRY_METHOD.md"),
              (ROOT / "build-support/authored-stocking-target/material-correspondence-audit.json", "checks/material-correspondence-audit.json")]
    for name in ("reference-tail-intersections.json", "tail-director-frames-audit.json",
                 "capsule-crease-cache-verification.json",
                 "left-verification.json", "right-verification.json", "both-verification.json",
                 "left-shared-edge-verification.json", "right-shared-edge-verification.json"):
        path = ROOT / "build-support/stocking-material/budget-features" / name
        if path.exists():
            extras.append((path, "checks/" + name))
    for name in ("right-three-states.png", "right-strain.png"):
        path = ROOT / "render/authored-fit-v1" / name
        if path.exists():
            extras.append((path, "preview/" + name))
    yarn_preview = ROOT / "render/authored-yarn-v1/matched-regular-window.png"
    if yarn_preview.exists():
        extras.append((yarn_preview, "preview/matched-regular-window.png"))
    plan = plan_authored_export(args.fit, args.features, maximum_bytes=args.maximum_bytes, extras=extras)
    print("AUTHORED_EXPORT_PLAN", json.dumps({key: value for key, value in plan.items() if key != "files"}))
    if args.write:
        report = write_authored_export(plan, args.output)
        print("AUTHORED_EXPORT_RESULT", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
