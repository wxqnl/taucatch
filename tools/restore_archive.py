"""Restore archived file paths without starting any job or changing monitor STOP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "archive/20261028/cleanup_manifest.json"
    moves = json.loads(manifest.read_text())["moves"]
    pending = []
    for entry in moves:
        original, archived = ROOT / entry["original"], ROOT / entry["archive"]
        if any(Path(entry[k]).is_absolute() or ".." in Path(entry[k]).parts
               for k in ["original", "archive"]):
            raise ValueError("Unsafe manifest path")
        if original.exists() or original.is_symlink():
            raise FileExistsError(f"Would overwrite: {original}")
        if entry.get("type") == "symlink":
            if not archived.is_symlink() or os.readlink(archived) != entry["target"]:
                raise ValueError(f"Archived link changed: {archived}")
        elif sha256(archived) != entry["sha256"]:
            raise ValueError(f"Archive changed: {archived}")
        pending.append((archived, original, entry.get("type")))
    if args.apply:
        for archived, original, kind in pending:
            original.parent.mkdir(parents=True, exist_ok=True)
            if kind == "symlink":
                original.symlink_to(os.readlink(archived))
            else:
                shutil.copy2(archived, original)
    print(json.dumps({"files": len(pending), "applied": args.apply,
                      "monitor_and_experiments_started": False}, indent=2))


if __name__ == "__main__":
    main()
