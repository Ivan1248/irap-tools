"""Extract all .rar archives in a directory into a per-video subdirectory layout.

Each archive contains entries like ``splitN/<video_dir>/<video_dir>_segNNN.png``.
The script strips the outer ``splitN/`` wrapper (an arbitrary RAR-partitioning
artifact) and writes files under ``<out>/<video_dir>/<basename>.png``, preserving
the per-video grouping. Same-basename files across different video directories
are not collisions; only two archives writing the **same post-strip path** count.
By default the script aborts on any such duplicate path; pass
``--ignore-duplicates`` to proceed – duplicates that share (size, CRC32) are
silently kept (first archive wins), while duplicates with differing content
trigger a warning.

The images output directory is wiped before extraction so that no stale or
partially-written files from a previous run survive. The script prompts for
confirmation when the directory is non-empty; pass ``--yes`` to skip the
prompt.

Requires ``unrar`` (preferred) or ``7z`` on the PATH.

Usage:
    python irap_vietnam_data_preparation/extract_images.py <data_dir>

Reads ``<data_dir>/_raw/image_rars/*.rar``; writes nested images into
``<data_dir>/images/<video_dir>/``.
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import layout


SPLIT_WRAPPER_RE = re.compile(r"^split\d+/", re.IGNORECASE)


def strip_split_wrapper(internal_path: str) -> str:
    """Strip a single leading ``splitN/`` component from an archive entry path."""
    return SPLIT_WRAPPER_RE.sub("", internal_path, count=1)


def find_extractor() -> tuple[str, str]:
    """Return (binary_name, kind) for the first available extractor."""
    for name in ("unrar", "unrar.exe", "UnRAR.exe"):
        if shutil.which(name):
            return name, "unrar"
    for name in ("7z", "7z.exe", "7zz"):
        if shutil.which(name):
            return name, "7z"
    raise RuntimeError(
        "Neither 'unrar' nor '7z' found on PATH. Install one to extract rar archives."
    )


def list_entries(binary: str, kind: str, archive: Path
                 ) -> list[tuple[str, int, str | None]]:
    """Return [(internal_path, size, crc32_hex), ...] for files in the archive.

    ``crc32_hex`` is the lowercased zero-padded hex string when the archive
    metadata records it, or ``None`` otherwise.
    """
    if kind == "unrar":
        out = subprocess.check_output([binary, "lt", str(archive)], text=True)
        entries: list[tuple[str, int, str | None]] = []
        name: str | None = None
        size: int | None = None
        crc: str | None = None
        is_dir = False
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Name:"):
                name = s[len("Name:"):].strip()
            elif s.startswith("Size:"):
                try:
                    size = int(s[len("Size:"):].strip())
                except ValueError:
                    size = None
            elif s.startswith("Type:"):
                is_dir = s[len("Type:"):].strip().lower() == "directory"
            elif s.startswith("CRC32:"):
                crc = s[len("CRC32:"):].strip().lower() or None
            elif not s and name is not None:
                if not is_dir and size is not None:
                    entries.append((name, size, crc))
                name, size, crc, is_dir = None, None, None, False
        if name is not None and not is_dir and size is not None:
            entries.append((name, size, crc))
        return entries

    # 7z
    out = subprocess.check_output([binary, "l", "-slt", str(archive)], text=True)
    entries = []
    path: str | None = None
    size: int | None = None
    crc: str | None = None
    is_dir = False
    for line in out.splitlines():
        if line.startswith("Path = "):
            path = line[len("Path = "):]
        elif line.startswith("Size = "):
            try:
                size = int(line[len("Size = "):])
            except ValueError:
                size = None
        elif line.startswith("CRC = "):
            crc = line[len("CRC = "):].strip().lower() or None
        elif line.startswith("Attributes = "):
            is_dir = "D" in line[len("Attributes = "):].split()[0]
        elif line == "" and path is not None:
            if not is_dir and size is not None and path != str(archive):
                entries.append((path, size, crc))
            path, size, crc, is_dir = None, None, None, False
    return entries


def _run_extraction(
        cmd: list[str], kind: str, total: int, desc: str, position: int = 0,
) -> None:
    """Run *cmd* (unrar/7z), drive a tqdm progress bar, raise on non-zero exit."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None

    bar = tqdm(total=total, unit="file", desc=desc,
               dynamic_ncols=True, position=position, leave=True)
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if kind == "unrar":
                if line.startswith("Extracting") or "is not extracted" in line:
                    bar.update(1)
                    continue
            else:
                if line.startswith("- "):
                    bar.update(1)
                    continue
            lo = line.lower()
            if line and ("error" in lo or "failed" in lo or "cannot" in lo
                         or "warning" in lo):
                bar.write(line)
    finally:
        rc = proc.wait()
        if bar.n < total and rc == 0:
            bar.update(total - bar.n)
        bar.close()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def extract(binary: str, kind: str, archive: Path, out: Path, *,
            num_entries: int, position: int = 0, overwrite: bool = False) -> None:
    """Extract one archive preserving per-video subfolders; show a tqdm bar.

    Uses ``unrar x`` / ``7z x`` so subdirectories from inside the archive are
    retained. The resulting ``<out>/splitN/`` wrapper directories are *not*
    flattened here – callers must invoke :func:`_flatten_split_wrappers` once
    after all archives are extracted (deferring is required so that concurrent
    extractions of archives sharing a ``splitN/<video_dir>/`` path do not race).

    Caller is expected to have wiped ``out`` beforehand; the extractor uses
    skip-if-exists mode (``-o-``/``-aos``) by default, or overwrite mode
    (``-y``/``-aoa``) if ``overwrite=True``.

    Drives the bar by parsing per-file lines from the extractor:
    - unrar: "Extracting  <path>" per file.
    - 7z (with -bb1): "- <path>" per file.
    Errors are passed through; ``CalledProcessError`` is raised on non-zero
    exit.

    ``position`` is the tqdm bar row, used to stack bars when multiple
    extractions run concurrently.
    """
    out.mkdir(parents=True, exist_ok=True)
    if kind == "unrar":
        skip_mode = "-y" if overwrite else "-o-"
        cmd = [binary, "x", skip_mode, "-y", str(archive), str(out) + "/"]
    else:
        # -bb1 emits one line per file; -bso0/-bsp0 silence summary/progress.
        skip_mode = "-aoa" if overwrite else "-aos"
        cmd = [binary, "x", skip_mode, "-bb1", "-bso1", "-bsp0",
               f"-o{out}", str(archive)]
    _run_extraction(cmd, kind, num_entries, archive.name, position)


def _flatten_split_wrappers(out: Path) -> None:
    """Move ``<out>/splitN/<child>`` up to ``<out>/<child>``; remove emptied dirs.

    Directory collisions (same ``<child>`` already present as a dir) are
    resolved by merging the source dir's contents into the destination dir
    recursively. File collisions resolve as first-wins (the existing file is
    kept and the duplicate from the splitN/ source is discarded).
    """
    for split_dir in sorted(out.iterdir()):
        if not (split_dir.is_dir() and re.fullmatch(r"split\d+", split_dir.name, re.IGNORECASE)):
            continue
        _merge_dir_into(split_dir, out)
        if not any(split_dir.iterdir()):
            split_dir.rmdir()


def _merge_dir_into(src: Path, dst: Path) -> None:
    """Move children of ``src`` into ``dst``, merging directories recursively."""
    for child in sorted(src.iterdir()):
        target = dst / child.name
        if not target.exists():
            child.rename(target)
        elif child.is_dir() and target.is_dir():
            _merge_dir_into(child, target)
            if not any(child.iterdir()):
                child.rmdir()
        elif child.is_file() and target.is_file():
            child.unlink()
        else:
            print(f"WARN: refusing to merge {child} into {target} "
                  f"(type mismatch)", file=sys.stderr)


def _compute_collisions(
        archive_entries: dict[Path, list[tuple[str, int, str | None]]],
) -> tuple[dict[str, list], dict[str, list], int]:
    """Return ``(by_name, collisions, total_bytes)``.

    ``by_name`` maps each post-strip relative path to a list of
    ``(archive, inner_path, size, crc32_hex)`` tuples across all archives.
    ``collisions`` is the sub-dict of ``by_name`` entries with more than one
    occurrence.
    """
    by_name: dict[str, list[tuple[Path, str, int, str | None]]] = defaultdict(list)
    total_bytes = 0
    for archive, entries in archive_entries.items():
        for inner, size, crc in entries:
            key = strip_split_wrapper(inner)
            by_name[key].append((archive, inner, size, crc))
            total_bytes += size
    collisions = {n: lst for n, lst in by_name.items() if len(lst) > 1}
    return dict(by_name), collisions, total_bytes


def _print_collision_group(
        group: dict[str, list], stream, *, label: str | None = None,
) -> None:
    """Print up to 50 entries from *group* with optional *label* prefix."""
    for name, lst in list(group.items())[:50]:
        if label is None:
            print(f"  {name}", file=stream)
        else:
            print(f"  {label}: {name}", file=stream)
        for archive, inner, size, crc in lst:
            crc_s = crc or "n/a"
            print(f"    in {archive.name}: {inner} "
                  f"({size} B, CRC32={crc_s})", file=stream)
    if len(group) > 50:
        print(f"  ... and {len(group) - 50} more", file=stream)


def _report_collisions(
        collisions: dict[str, list],
        ignore_duplicates: bool) -> None:
    """Print a collision report; abort with ``SystemExit(2)`` unless *ignore_duplicates*."""
    differing: dict[str, list] = {}
    ambiguous: dict[str, list] = {}
    identical: dict[str, list] = {}
    for name, lst in collisions.items():
        sizes = {size for _, _, size, _ in lst}
        crcs = {crc for _, _, _, crc in lst if crc is not None}
        any_missing_crc = any(crc is None for _, _, _, crc in lst)
        if len(sizes) > 1 or len(crcs) > 1:
            differing[name] = lst
        elif any_missing_crc:
            ambiguous[name] = lst
        else:
            identical[name] = lst

    if not ignore_duplicates:
        print(f"ERROR: {len(collisions)} duplicate path(s) across archives "
              f"(pass --ignore-duplicates to proceed):", file=sys.stderr)
        _print_collision_group(collisions, sys.stderr)
        raise SystemExit(2)

    print(f"  {len(collisions)} duplicate path(s); ignoring "
          f"(first archive wins on extraction):")
    print(f"    identical content: {len(identical)}")
    print(f"    same size, CRC unavailable in >=1 archive: {len(ambiguous)}")
    print(f"    differing content: {len(differing)}")

    for label, group, stream in (("WARN [content differs]", differing, sys.stderr),
                                 ("note [size match, no CRC]", ambiguous, sys.stdout),
                                 ("note [identical content]", identical, sys.stdout)):
        if not group:
            continue
        _print_collision_group(group, stream, label=label)


def extract_specific(
        binary: str, kind: str, archive: Path, out: Path,
        entries: list[tuple[str, int, str | None]], *, position: int = 0,
) -> None:
    """Extract specific *entries* from *archive* into *out*, showing a tqdm bar."""
    if not entries:
        return
    out.mkdir(parents=True, exist_ok=True)
    internal_paths = [e[0] for e in entries]
    if kind == "unrar":
        cmd = [binary, "x", "-o-", "-y", str(archive), str(out) + "/"] + internal_paths
    else:
        cmd = [binary, "x", "-aos", "-bb1", "-bso1", "-bsp0",
               f"-o{out}", str(archive)] + internal_paths
    _run_extraction(cmd, kind, len(entries), f"[dup] {archive.name}", position)


def extract_duplicates(
        binary: str, kind: str,
        archive_entries: dict[Path, list[tuple[str, int, str | None]]],
        collisions: dict[str, list],
        dup_out: Path,
) -> None:
    """Extract all copies of duplicate entries into ``dup_out/<archive_stem>/``.

    For every archive that contributes at least one entry involved in a
    collision its duplicate entries are extracted to
    ``dup_out/<archive.stem>/`` and the ``splitN/`` wrapper is then flattened,
    yielding ``dup_out/<archive_stem>/<video_dir>/<file>.png``.
    """
    dup_inner_paths: dict[Path, set[str]] = defaultdict(set)
    for name, lst in collisions.items():
        for archive, inner, size, crc in lst:
            dup_inner_paths[archive].add(inner)

    for i, (archive, dup_paths) in enumerate(sorted(dup_inner_paths.items())):
        entries = [(inner, size, crc)
                   for inner, size, crc in archive_entries[archive]
                   if inner in dup_paths]
        archive_dup_out = dup_out / archive.stem
        extract_specific(binary, kind, archive, archive_dup_out, entries,
                         position=i)
        _flatten_split_wrappers(archive_dup_out)


def _count_files_in_dir(path: Path) -> int:
    """Count all files recursively in a directory."""
    return sum(1 for _ in path.rglob("*") if _.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    parser.add_argument("--ignore-duplicates", action="store_true",
                        help="Proceed when archives share post-strip paths "
                             "(first archive wins on extraction).")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt before wiping a "
                             "non-empty images directory.")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Parallel extractions. Default: "
                             "min(num_archives, max(4, cpu_count()//4)).")
    args = parser.parse_args(argv)

    rars = layout.rars_dir(args.data_dir)
    out = layout.images_dir(args.data_dir)
    all_archives = sorted(rars.glob("*.rar")) if rars.is_dir() else []
    if not all_archives:
        print(f"No .rar archives found in {rars}", file=sys.stderr)
        return 1

    # Separate regular archives from missing_segments archives
    regular_archives = [a for a in all_archives if "missing_segments" not in a.name.lower()]
    missing_archives = [a for a in all_archives if "missing_segments" in a.name.lower()]

    binary, kind = find_extractor()
    print(f"Using extractor: {binary} ({kind})")
    print(f"Found {len(all_archives)} archive(s):")
    if regular_archives:
        print(f"  Regular archives ({len(regular_archives)}):")
        for a in regular_archives:
            print(f"    {a.name}  ({a.stat().st_size / 1e9:.2f} GB)")
    if missing_archives:
        print(f"  Missing segments archives ({len(missing_archives)}):")
        for a in missing_archives:
            print(f"    {a.name}  ({a.stat().st_size / 1e9:.2f} GB)")

    # Index only regular archives for collision detection (missing_segments are meant to replace)
    print(f"Indexing {len(regular_archives)} regular archive(s)...")
    archive_entries: dict[Path, list[tuple[str, int, str | None]]] = {}
    for a in regular_archives:
        archive_entries[a] = list_entries(binary, kind, a)
        print(f"  {a.name}: {len(archive_entries[a])} entries")

    dup_out = layout.images_duplicates_dir(args.data_dir)

    # Stage 1: Extract regular archives (skip-if-exists)
    if regular_archives:
        if out.exists() and any(out.iterdir()):
            if not args.yes:
                print(f"\n{out} is non-empty and will be wiped before extraction.")
                reply = input("Proceed? [y/N] ").strip().lower()
                if reply not in ("y", "yes"):
                    print("Aborted.", file=sys.stderr)
                    return 1
            print(f"Wiping {out}...")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        num_jobs = args.jobs or min(len(regular_archives), 4)
        print(f"\n[Stage 1] Extracting {len(regular_archives)} regular archive(s) "
              f"with {num_jobs} parallel worker(s)...")

        with ThreadPoolExecutor(max_workers=num_jobs) as ex:
            futures = {
                ex.submit(extract, binary, kind, a, out,
                          num_entries=len(archive_entries[a]),
                          position=i, overwrite=False): a
                for i, a in enumerate(regular_archives)
            }
            for fut in as_completed(futures):
                fut.result()

        print("Flattening splitN/ wrappers...")
        _flatten_split_wrappers(out)
        print(f"Stage 1 complete. {sum(1 for _ in out.rglob('*') if _.is_file())} "
              f"file(s) in {out}")

    # Stage 2: Extract missing_segments archives (with overwrite)
    if missing_archives:
        print(f"\n[Stage 2] Extracting {len(missing_archives)} missing_segments "
              f"archive(s) with overwrite...")
        
        if not out.exists():
            out.mkdir(parents=True, exist_ok=True)
        
        num_jobs = args.jobs or min(len(missing_archives), 4)
        
        # Extract to temporary directory first
        temp_extract_dir = out.parent / f".temp_missing_extract_{id(out)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
        with ThreadPoolExecutor(max_workers=num_jobs) as ex:
            futures = {
                ex.submit(extract, binary, kind, a, temp_extract_dir,
                          num_entries=len(archive_entries[a]) if a in archive_entries else 0,
                          position=i, overwrite=True): a
                for i, a in enumerate(missing_archives)
            }
            for fut in as_completed(futures):
                fut.result()

        # Flatten wrappers in temp directory
        _flatten_split_wrappers(temp_extract_dir)
        
        # Count files to be merged (all will be replaced unconditionally)
        files_to_merge = _count_files_in_dir(temp_extract_dir)
        
        # Merge temp extracted files into main output, overwriting unconditionally
        print("Merging missing_segments files with overwrite...")
        for src_file in temp_extract_dir.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(temp_extract_dir)
                dst_file = out / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        
        # Cleanup temp directory
        shutil.rmtree(temp_extract_dir)
        
        print(f"Stage 2 complete. {files_to_merge} file(s) replaced or added.")
        print(f"Total file(s) in {out}: "
              f"{sum(1 for _ in out.rglob('*') if _.is_file())}")
    
    # Stage 3: Check for duplicates in regular archives only (not including missing_segments)
    print("\n[Stage 3] Checking for duplicates in regular archives only...")
    if archive_entries:
        by_name, collisions, total = _compute_collisions(archive_entries)
        num_files = len(by_name)
        print(f"  {num_files} unique basenames, {total / 1e9:.2f} GB total "
              f"(uncompressed)")

        if collisions:
            if dup_out.exists():
                shutil.rmtree(dup_out)
            dup_out.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {len(collisions)} duplicate path(s) to {dup_out}...")
            extract_duplicates(binary, kind, archive_entries, collisions, dup_out)
            dup_count = sum(1 for p in dup_out.rglob("*") if p.is_file())
            print(f"  {dup_count} file(s) written to {dup_out}")
            _report_collisions(collisions, ignore_duplicates=args.ignore_duplicates)
        else:
            print("  No duplicates found in regular archives.")
    else:
        print("  No regular archives to check for duplicates.")
    
    print(f"\nDone. {sum(1 for _ in out.rglob('*') if _.is_file())} file(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
