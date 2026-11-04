#!/usr/bin/env python3
"""Import local dependency ZIPs for the inspected 2026 TFLM snapshot.

Uses Python's standard library and TFLM's own verification/preparation code.
No network calls, pip installs, compiler changes, or source patches.
Run from framework_vendor, with the three ZIPs in offline_deps/:
    python3 prepare_tflm_offline.py
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys

DOWNLOADER_SHA256 = "86438a6bd22b7ede7137afe1cc22ca5910e636734e7303d46efed44c038bc01f"


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_downloader(root):
    path = root / "tensorflow/lite/micro/tools/make/download_dependencies.py"
    if not path.is_file():
        raise RuntimeError("TFLM source not found: " + str(path))
    if digest(path, "sha256") != DOWNLOADER_SHA256:
        raise RuntimeError("Different TFLM downloader version. Use the inspected source archive.")
    spec = importlib.util.spec_from_file_location("tflm_offline_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(module, root, archives):
    downloads = root / "tensorflow/lite/micro/tools/make/downloads"
    pending = []
    for name in module.DEFAULT_DEPENDENCIES:
        dep = module.DEPENDENCIES[name]
        if module.is_already_downloaded(dep, downloads / name, root):
            print("Already prepared: " + name, flush=True)
        else:
            pending.append(dep)

    # Preflight every archive before changing the dependency tree.
    by_hash = {}
    if archives.is_dir():
        for path in sorted(archives.glob("*.zip")):
            by_hash.setdefault(digest(path, "md5"), path)
    local_sources = {}
    missing = []
    for dep in pending:
        if dep.type != "archive" or not dep.md5:
            raise RuntimeError("Unsupported dependency definition: " + dep.name)
        source = by_hash.get(dep.md5)
        if source is None:
            missing.append("{}: expected MD5 {}\n  {}".format(dep.name, dep.md5, dep.url))
        else:
            local_sources[dep.url] = source
    if missing:
        raise RuntimeError("Missing or incorrect ZIP(s) in {}:\n{}".format(
            archives, "\n".join(missing)))

    def copy_local(url, output_path, max_retries=5):
        source = local_sources.get(url)
        if source is None:
            raise RuntimeError("Offline mode: no verified local archive for " + url)
        shutil.copyfile(str(source), str(output_path))

    # Only the transport is replaced. Official hash checks, extraction, BUILD
    # cleanup, metadata stamps, and atomic installation remain in use.
    original_transport = module.download_url_with_retry
    module.download_url_with_retry = copy_local
    try:
        for dep in pending:
            print("Preparing: {} ({})".format(dep.name, local_sources[dep.url].name), flush=True)
            module.download_and_prepare(dep, downloads, root)
    finally:
        module.download_url_with_retry = original_transport

    for name in module.DEFAULT_DEPENDENCIES:
        if not module.is_already_downloaded(module.DEPENDENCIES[name], downloads / name, root):
            raise RuntimeError("Completion check failed: " + name)
    print("SUCCESS: all 3 TFLM dependencies are ready offline.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=base / "tflite-micro-main")
    parser.add_argument("--archives", type=Path, default=base / "offline_deps")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    archives = args.archives.expanduser().resolve()
    try:
        prepare(load_downloader(root), root, archives)
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
