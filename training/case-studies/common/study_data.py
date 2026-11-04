"""Restore a versioned case-study bundle without network or inference calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_data(study: str, root: Path | None = None) -> Path:
    root = (root or Path(__file__).resolve().parent.parent).resolve()
    catalog = json.loads((root / "bundles/manifest.json").read_text())
    if catalog["version"] != 1:
        raise ValueError("Unsupported bundle manifest version")
    if study not in catalog["bundles"]:
        raise ValueError(f"Unknown study: {study}")
    entry = catalog["bundles"][study]
    archive = root / "bundles" / entry["archive"]
    if not archive.is_file():
        raise FileNotFoundError(
            f"Download {entry['archive']} from the cookbook's training/case-studies/bundles "
            f"directory and save it at {archive}. Use the same revision as manifest.json."
        )
    data = archive.read_bytes()
    if digest(data) != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {archive.name}. Download the matching revision again.")
    # Validate the entire archive and all existing targets before writing any files.
    import io
    pending = []
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(entry["files"]):
            raise ValueError("Archive contents do not match the manifest")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"Unsafe bundle path: {name}")
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root) or any(
                p.is_symlink() for p in [target, *target.parents] if p != root
            ):
                raise ValueError(f"Unsafe extraction target: {name}")
            content = bundle.read(name)
            if digest(content) != entry["files"][name]:
                raise ValueError(f"File checksum mismatch: {name}")
            if target.exists():
                if not target.is_file() or digest(target.read_bytes()) != entry["files"][name]:
                    raise FileExistsError(f"Keeping your changed file: {target}. Move it before restoring data.")
            else:
                pending.append((target, content))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also avoids replacing a file created after validation.
        with target.open("xb") as stream:
            stream.write(content)
    print(f"{study}: verified {len(entry['files'])} files; restored {len(pending)}.")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", help="caveman, humanizer, anti-slop, e-vasion, taste, ask-before-acting")
    args = parser.parse_args()
    ensure_data(args.study)


if __name__ == "__main__":
    main()
