"""Inspect, verify and package the curated handoff; never launches experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular file: {path}")
    return path


def verify(root):
    count = 0
    for line in (root / "artifacts/MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha256(local(root, name)) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        count += 1
    sources = json.loads((root / "artifacts/source_manifest.json").read_text())
    for entry in sources["files"]:
        if sha256(local(root, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Source mismatch: {entry['path']}")
    print(json.dumps({"accepted": True, "checksummed_files": count,
                      "source_files": len(sources["files"]),
                      "new_experiments": 0}, indent=2))


def refresh(root):
    """Refresh the existing handoff allowlist after intentional source/media edits."""
    manifest = root / "artifacts/MANIFEST.sha256"
    names = {line.split("  ", 1)[1] for line in manifest.read_text().splitlines()}
    selection = json.loads((root / "artifacts/source_selection.json").read_text())
    source = set(selection["dependency_closure"])
    source.update(str(p.relative_to(root)) for p in (root / "tools").glob("*.py"))
    entries = [{"path": name, "sha256": sha256(local(root, name)),
                "bytes": local(root, name).stat().st_size} for name in sorted(source)]
    (root / "artifacts/source_manifest.json").write_text(json.dumps({
        "scope": "Curated implementation and handoff tools; source bytes identified by SHA256",
        "files": entries}, indent=2) + "\n")
    names.update(source)
    manifest.write_text("".join(f"{sha256(local(root, name))}  {name}\n"
                                 for name in sorted(names)))
    print(json.dumps({"refreshed_files": len(names), "source_files": len(source)}))


def results(root):
    completion = json.loads((root / "artifacts/evidence/generalization/completion.json").read_text())
    rows = completion["summary"]["pairs"]
    assert len(rows) == 6 and completion["physical_trials"] == 12
    with (root / "artifacts/results.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 12
    assert {(row["seed"], row["mode"]) for row in exported} == {
        (seed, mode) for seed in rows for mode in ["VLA", "fast_only"]}
    for row in exported:
        pair = rows[row["seed"]]
        key = "mean_held" if row["mode"] == "VLA" else "zero_held"
        assert int(row["held"]) == int(pair[key])
        assert float(row["speed_mps"]) == pair["speed_mps"]
    result = {"scope": "Six development geometries, paired simulation; not final test",
              "VLA": {"held": sum(p["mean_held"] for p in rows.values()), "trials": 6},
              "fast_only": {"held": sum(p["zero_held"] for p in rows.values()), "trials": 6}}
    flight = [r for p in completion["parts"] for r in p["requests_summary"] if r["op"] == "fast"]
    roundtrip = sorted(r["roundtrip_ms"] for r in flight)
    ready = sorted((r["available_t"] - r["observed_t"]) * 1e3 for r in flight)
    result["flight_latency_ms"] = {
        "requests": len(flight),
        "roundtrip": {"median": round(statistics.median(roundtrip), 2),
                      "min": round(roundtrip[0], 2), "max": round(roundtrip[-1], 2)},
        "observation_to_available": {"median": round(statistics.median(ready), 2),
                                     "min": round(ready[0], 2), "max": round(ready[-1], 2)}}
    print(json.dumps(result, indent=2))


def build(root, output):
    verify(root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = root / "artifacts/MANIFEST.sha256"
    paths = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
    paths.append("artifacts/MANIFEST.sha256")
    target = output / "spikevla-catch-source.tar.gz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    with tarfile.open(target, "w:gz", compresslevel=6, dereference=True) as archive:
        for name in sorted(set(paths)):
            archive.add(local(root, name), arcname="spikevla-catch/" + name,
                        recursive=False)
    print(json.dumps({"source_archive": str(target), "bytes": target.stat().st_size,
                      "sha256": sha256(target)}, indent=2))


def build_weights(root, output):
    entries = json.loads((root / "artifacts/checkpoints.json").read_text())["files"]
    for entry in entries:
        if sha256(local(root, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {entry['path']}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "spikevla-catch-checkpoints.tar.gz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    with tarfile.open(target, "w:gz", compresslevel=3, dereference=True) as archive:
        for name in [e["path"] for e in entries] + ["artifacts/checkpoints.json"]:
            archive.add(local(root, name), arcname="spikevla-catch/" + name,
                        recursive=False)
    print(json.dumps({"checkpoint_archive": str(target), "bytes": target.stat().st_size,
                      "sha256": sha256(target)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "refresh", "results", "build", "build-weights"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "verify":
        verify(root)
    elif args.command == "refresh":
        refresh(root)
    elif args.command == "results":
        results(root)
    elif args.command == "build":
        build(root, args.output)
    else:
        build_weights(root, args.output)


if __name__ == "__main__":
    main()
