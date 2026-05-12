"""Download all files from a Seafile public share link.

Defaults to the IRAP Vietnam dataset share. Supports resume via HTTP Range and
skips files whose local size already matches the remote size. Recurses into
subdirectories.

Usage:
    python irap_vietnam_data_preparation/download_images.py <data_dir>

Writes to ``<data_dir>/_raw/image_rars/``. Use ``--share-url`` for a non-default share.
"""
import argparse
import json
import re
import sys
import typing as T
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

import layout

DEFAULT_SHARE_URL = "https://seafile.irap.org/d/074e5d6b7d1047b998a7/"
CHUNK_SIZE = 1 << 20  # 1 MiB


def parse_share_url(share_url: str) -> tuple[str, str]:
    """Return (server_root, token) from a share link like https://host/d/<token>/."""
    m = re.match(r"^(https?://[^/]+)/d/([0-9a-f]+)/?", share_url)
    if not m:
        raise ValueError(f"Not a Seafile share link: {share_url!r}")
    return m.group(1), m.group(2)


def list_dirents(server: str, token: str, path: str = "/",
                 password: str | None = None) -> list[dict]:
    """List entries directly under `path` on the share."""
    qs = {"path": path}
    if password:
        qs["password"] = password
    url = f"{server}/api/v2.1/share-links/{token}/dirents/?{urllib.parse.urlencode(qs)}"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    return data["dirent_list"]


def walk_files(server: str, token: str, password: str | None = None
               ) -> T.Iterator[dict]:
    """Yield file dirents from the share, recursing into folders."""
    stack = ["/"]
    while stack:
        path = stack.pop()
        for entry in list_dirents(server, token, path, password):
            if entry["is_dir"]:
                stack.append(entry["folder_path"])
            else:
                yield entry


def download_file(server: str, token: str, remote_path: str, dest: Path,
                  expected_size: int, password: str | None = None) -> None:
    """Download `remote_path` to `dest`, resuming if a partial file exists."""
    qs = {"p": remote_path, "dl": "1"}
    if password:
        qs["password"] = password
    url = f"{server}/d/{token}/files/?{urllib.parse.urlencode(qs)}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing == expected_size:
        print(f"[skip] {remote_path} ({expected_size} bytes already present)")
        return
    if existing > expected_size:
        # Local file is larger than remote – likely corrupt; restart.
        print(f"[warn] {dest} larger than remote ({existing} > {expected_size}); "
              f"restarting download")
        dest.unlink()
        existing = 0

    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[resume] {remote_path} from byte {existing}")

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416 and existing == expected_size:
            return  # Already complete.
        raise

    with resp, open(dest, mode) as f:
        with tqdm(
                total=expected_size, initial=existing, unit="B", unit_scale=True,
                unit_divisor=1024, desc=remote_path.lstrip("/"), miniters=1,
        ) as bar:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    final = dest.stat().st_size
    if final != expected_size:
        raise RuntimeError(
            f"Size mismatch for {remote_path}: got {final}, expected {expected_size}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    args = parser.parse_args(argv)

    server, token = parse_share_url(DEFAULT_SHARE_URL)
    out = layout.rars_dir(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = list(walk_files(server, token, None))
    total = sum(f["size"] for f in files)
    print(f"Found {len(files)} file(s), total {total / 1e9:.2f} GB")
    for f in files:
        print(f"  {f['size']:>14}  {f['file_path']}")

    for f in files:
        rel = f["file_path"].lstrip("/")
        dest = out / rel
        download_file(server, token, f["file_path"], dest, f["size"], None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
