"""Extract all .rar archives in a directory into a per-video subdirectory layout.

Each archive contains entries like ``splitN/<video_dir>/<video_dir>_segNNN.png``.
The script strips the outer ``splitN/`` wrapper (an arbitrary RAR-partitioning
artifact) and writes files under ``FRAMES/<video_dir>/<basename>.png``, preserving
the per-video grouping. Same-basename files across different video directories
are not collisions; only two archives writing the **same post-strip path** count.
By default the script aborts on any such duplicate path. With
``--ignore-duplicates`` it proceeds and keeps one copy of each path, which is
not necessarily the one of the first archive. It warns about the paths whose
copies differ in size or CRC32.

Regular archives are named ``split<N>.rar``. Patch archives hold corrected,
renamed or additional segment images. They are listed in
:data:`PATCH_ARCHIVE_ORDER`, in the order in which they are applied, after the
regular archives, unconditionally replacing the image of the same seg_id, so
that a later patch archive wins. An image of that seg_id at another path than
the patch file's name gives is removed (see :func:`plan_patch_merge`). They do
not participate in the duplicate check (see :func:`_apply_patch_archives`). An
archive with another name, or a missing patch archive, stops the script.

The images output directory is wiped before extraction so that no stale or
partially-written files from a previous run survive. The script prompts for
confirmation when the directory is non-empty; pass ``--yes`` to skip the
prompt. With ``--patches-only`` the script keeps the directory and only applies
the patch archives to it, e.g. after an interrupted patch step or when a patch
archive is added.

Requires ``unrar`` (preferred) or ``7z`` on the PATH.

Usage:
    python extract_images.py <data_dir> [--ignore-duplicates] [--patches-only]

Reads ``<data_dir>/_raw/image_rars/*.rar``; writes nested images into
``<data_dir>/FRAMES/<video_dir>/``.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import typing as T
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import layout


SPLIT_DIR_NAME_RE = re.compile(r"split\d+", re.IGNORECASE)
SPLIT_WRAPPER_RE = re.compile(rf"^{SPLIT_DIR_NAME_RE.pattern}/", re.IGNORECASE)

# Matches the name of a regular archive.
REGULAR_ARCHIVE_NAME_RE = re.compile(rf"{SPLIT_DIR_NAME_RE.pattern}\.rar", re.IGNORECASE)

# The patch archives, whose images replace or add to those of the regular archives, in
# the order in which they are applied: for a file in several patch archives, the last
# one wins.
PATCH_ARCHIVE_ORDER = (
    "missing_segments1_4.rar",  # 2026-06-04
    "missing_segments5_7.rar",  # 2026-06-05
    "new files.rar",  # 2026-11-06
    "Corrected_sequences.rar",  # 2026-11-10
    "Corrected_sequences2.rar",  # 2026-11-10, later in the day
    "UpdatedData-24Nov2026.rar",  # 2026-11-24
)


def strip_split_wrapper(internal_path: str) -> str:
    """Strip a single leading ``splitN/`` component from an archive entry path."""
    return SPLIT_WRAPPER_RE.sub("", internal_path, count=1)


def split_regular_and_patch_archives(archives: list[Path]) -> tuple[list[Path], list[Path]]:
    """Returns the regular archives and the patch archives in `PATCH_ARCHIVE_ORDER`.

    Raises ValueError if an archive is neither a regular nor a patch archive, or if a
    patch archive is missing, as both would silently change which images are kept.
    """
    name_to_archive = {a.name: a for a in archives}
    regular = [a for a in archives if REGULAR_ARCHIVE_NAME_RE.fullmatch(a.name)]
    unknown = [a.name for a in archives
               if a not in regular and a.name not in PATCH_ARCHIVE_ORDER]
    if unknown:
        raise ValueError(f"Unknown archives {unknown}: a regular archive is named"
                         f" split<N>.rar, and a patch archive must be added to"
                         f" PATCH_ARCHIVE_ORDER.")
    missing = [name for name in PATCH_ARCHIVE_ORDER if name not in name_to_archive]
    if missing:
        raise ValueError(f"Missing patch archives {missing}.")
    return regular, [name_to_archive[name] for name in PATCH_ARCHIVE_ORDER]


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


def _index_archives(binary: str, kind: str, archives: list[Path]
                   ) -> dict[Path, list[tuple[str, int, str | None]]]:
    """Return ``{archive: list_entries(...)}``, printing each archive's entry count."""
    entries: dict[Path, list[tuple[str, int, str | None]]] = {}
    for a in archives:
        entries[a] = list_entries(binary, kind, a)
        print(f"  {a.name}: {len(entries[a])} entries")
    return entries


def _extract_cmd(binary: str, kind: str, archive: Path, out: Path,
                 members: list[str] | None = None) -> list[str]:
    """Build the unrar/7z command extracting *archive* (or just *members*) into *out*."""
    if kind == "unrar":
        cmd = [binary, "x", "-o-", "-y", str(archive), str(out) + "/"]
    else:
        # -bb1 emits one line per file; -bso0/-bsp0 silence summary/progress.
        cmd = [binary, "x", "-aos", "-bb1", "-bso1", "-bsp0",
               f"-o{out}", str(archive)]
    return cmd + (members or [])


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
            num_entries: int, position: int = 0) -> None:
    """Extract one archive preserving per-video subfolders; show a tqdm bar.

    Uses ``unrar x`` / ``7z x`` so subdirectories from inside the archive are
    retained. The resulting ``<out>/splitN/`` wrapper directories are *not*
    flattened here – callers must invoke :func:`_flatten_split_wrappers` once
    after all archives are extracted (deferring is required so that concurrent
    extractions of archives sharing a ``splitN/<video_dir>/`` path do not race).

    Caller is expected to have wiped ``out`` beforehand; the extractor still
    uses skip-if-exists mode (``-o-``/``-aos``) so that two workers writing
    to the same ``splitN/<video_dir>/`` path resolve to first-wins instead of
    racing on a partial overwrite.

    Drives the bar by parsing per-file lines from the extractor:
    - unrar: "Extracting  <path>" per file.
    - 7z (with -bb1): "- <path>" per file.
    Errors are passed through; ``CalledProcessError`` is raised on non-zero
    exit.

    ``position`` is the tqdm bar row, used to stack bars when multiple
    extractions run concurrently.
    """
    out.mkdir(parents=True, exist_ok=True)
    _run_extraction(_extract_cmd(binary, kind, archive, out),
                    kind, num_entries, archive.name, position)


def _flatten_split_wrappers(out: Path) -> None:
    """Move ``<out>/splitN/<child>`` up to ``<out>/<child>``; remove emptied dirs.

    Directory collisions (same ``<child>`` already present as a dir) are
    resolved by merging the source dir's contents into the destination dir
    recursively. File collisions resolve as first-wins (the existing file is
    kept and the duplicate from the splitN/ source is discarded).
    """
    for split_dir in sorted(out.iterdir()):
        if not (split_dir.is_dir() and SPLIT_DIR_NAME_RE.fullmatch(split_dir.name)):
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


def _print_collision_group(group: dict[str, list], stream, *, label: str) -> None:
    """Print up to 50 entries from *group*, each prefixed with *label*."""
    for name, lst in list(group.items())[:50]:
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
    """Print a collision report; abort with ``SystemExit(2)`` unless *ignore_duplicates*.

    The copies of a path are identical if they have equal size and CRC32.
    """
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

    print(f"  {len(collisions)} duplicate path(s) across archives:")
    print(f"    identical content: {len(identical)}")
    print(f"    same size, CRC unavailable in >=1 archive: {len(ambiguous)}")
    print(f"    differing content: {len(differing)}")

    for label, group, stream in (("WARN [content differs]", differing, sys.stderr),
                                 ("note [size match, no CRC]", ambiguous, sys.stdout),
                                 ("note [identical content]", identical, sys.stdout)):
        if not group:
            continue
        _print_collision_group(group, stream, label=label)

    if not ignore_duplicates:
        print(f"ERROR: {len(collisions)} duplicate path(s) across archives "
              f"(pass --ignore-duplicates to keep one copy of each).", file=sys.stderr)
        raise SystemExit(2)

    print("  Ignoring the duplicates: one copy of each path is kept.")


def extract_specific(
        binary: str, kind: str, archive: Path, out: Path,
        entries: list[tuple[str, int, str | None]], *, position: int = 0,
) -> None:
    """Extract specific *entries* from *archive* into *out*, showing a tqdm bar."""
    if not entries:
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = _extract_cmd(binary, kind, archive, out, [e[0] for e in entries])
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


def _extract_parallel(
        binary: str, kind: str, archives: list[Path],
        archive_entries: dict[Path, list[tuple[str, int, str | None]]],
        dest_for: Callable[[Path], Path], *, jobs: int | None, label: str,
) -> None:
    """Extract *archives* concurrently, each into ``dest_for(archive)``."""
    num_jobs = jobs or min(len(archives), 4)
    print(f"\nExtracting {len(archives)} {label} archive(s) with "
          f"{num_jobs} parallel worker(s)...")
    with ThreadPoolExecutor(max_workers=num_jobs) as ex:
        futures = {
            ex.submit(extract, binary, kind, a, dest_for(a),
                      num_entries=len(archive_entries[a]),
                      position=i): a
            for i, a in enumerate(archives)
        }
        for fut in as_completed(futures):
            fut.result()


def _print_path_examples(paths: list[Path], root: Path) -> None:
    for p in paths[:20]:
        print(f"    {p.relative_to(root)}", file=sys.stderr)
    if len(paths) > 20:
        print(f"    ... and {len(paths) - 20} more", file=sys.stderr)


@dataclass(frozen=True)
class PatchMerge:
    """How the extracted patch files are merged into the images directory."""

    moves: list[tuple[Path, Path]]  # (extracted patch file, destination), in merge order
    # The images that are removed after the moves, each with the destination of the
    # last patch file of its seg_id, which is another path.
    superseded: dict[Path, Path]
    unmatched: list[Path]  # patch files not named <video_dir>_seg<seg_id>.png


def plan_patch_merge(
        patch_files: T.Sequence[Path], out: Path,
        seg_id_to_images: T.Mapping[str, T.Collection[Path]],
) -> PatchMerge:
    """Plans the merge of the extracted *patch_files*, in order, into the images directory *out*.

    A patch file becomes the image of its seg_id at ``out/<video_dir>/<file name>``,
    with a lowercase extension. Every other image of that seg_id is superseded:
    those in *seg_id_to_images* (the images in *out*) and those of earlier patch
    files with another name. Each seg_id of a patch file thus keeps a single
    image, at the path given by its last patch file, so the result does not
    depend on the patch files that an earlier run merged. A patch archive that
    names the folder of a recording differently renames it, e.g.
    ``_20241211_140925_00_004`` to ``VID_20241211_140925_00_004``.
    """
    seg_id_to_paths = {s: set(paths) for s, paths in seg_id_to_images.items()}
    seg_id_to_destination: dict[str, Path] = {}
    moves: list[tuple[Path, Path]] = []
    unmatched: list[Path] = []
    for patch_file in patch_files:
        seg_id = layout.seg_id_from_image_name(patch_file.name)
        if seg_id is None:
            unmatched.append(patch_file)
            continue
        video_dir = layout.video_dir_from_image_name(patch_file.name)
        # The images are found by "*.png", which does not match ".PNG" on Linux.
        destination = out / video_dir / patch_file.with_suffix(".png").name
        moves.append((patch_file, destination))
        seg_id_to_paths.setdefault(seg_id, set()).add(destination)
        seg_id_to_destination[seg_id] = destination
    superseded = {image: destination
                  for seg_id, destination in seg_id_to_destination.items()
                  for image in sorted(seg_id_to_paths[seg_id] - {destination})}
    return PatchMerge(moves, superseded, unmatched)


def _apply_patch_archives(
        binary: str, kind: str, archives: list[Path], out: Path, tmp: Path,
        *, jobs: int | None,
) -> None:
    """Extract patch *archives* and merge their files into ``out``.

    Each archive is extracted into its own fresh subdirectory of the scratch
    directory ``tmp`` (fresh, so the extractor's skip-if-exists mode never
    triggers and concurrent extractions cannot race). Every
    ``<video_dir>_segN.png`` file then replaces the image of its seg_id: it is
    moved to ``out/<video_dir>/<file>``, and the images of its seg_id at other
    paths are removed (see :func:`plan_patch_merge`). Their folders are reported.
    The owning ``<video_dir>`` is recovered from the filename because some patch
    archives wrap the video folders in arbitrary (sometimes nested) folders, so
    the extracted directory structure cannot be trusted the way the regular
    ``splitN/`` archives' can. Files not matching the naming convention are
    reported and skipped. The archives are merged in the order of *archives*, so
    for a seg_id in several patch archives the last archive wins. Such files are
    reported.
    """
    print(f"\nIndexing {len(archives)} patch archive(s)...")
    entries = _index_archives(binary, kind, archives)

    def get_extraction_dir(archive: Path) -> Path:
        return tmp / archive.stem

    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        _extract_parallel(binary, kind, archives, entries, get_extraction_dir, jobs=jobs,
                          label="patch")

        # Moving a file whose data is not on the disk yet can wait for that data, which
        # takes much longer file by file than for all the files at once.
        if hasattr(os, "sync"):  # not available on Windows
            print("Writing the extracted patch files to the disk...")
            os.sync()

        print("Merging patch files into the images directory...")
        src_files = [f for a in archives for f in sorted(get_extraction_dir(a).rglob("*"))
                     if f.is_file()]
        merge = plan_patch_merge(src_files, out, layout.index_images_by_seg_id(out))
        replaced = 0  # overwrote the file at its destination
        added = 0     # its destination is a new path
        patched_files: set[Path] = set()
        files_in_several_archives: set[Path] = set()
        for src_file, dst_file in tqdm(merge.moves, unit="file", desc="Merging",
                                       dynamic_ncols=True):
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file in patched_files:
                files_in_several_archives.add(dst_file)
            elif dst_file.exists():
                replaced += 1
            else:
                added += 1
            patched_files.add(dst_file)
            src_file.replace(dst_file)
        for image in merge.superseded:
            image.unlink()
        for folder in {image.parent for image in merge.superseded}:
            if not any(folder.iterdir()):
                folder.rmdir()

        if merge.unmatched:
            print(f"WARN: {len(merge.unmatched)} patch file(s) did not match "
                  f"the <video_dir>_segN.png naming convention and were skipped:",
                  file=sys.stderr)
            _print_path_examples(merge.unmatched, tmp)

        if merge.superseded:
            folder_pair_counts = Counter((image.parent, dst.parent)
                                         for image, dst in merge.superseded.items())
            print(f"WARN: {len(merge.superseded)} image(s) were removed, since a patch "
                  f"file replaced them at another path (folder of the removed images -> "
                  f"folder of the patch files: number of images, files left in the "
                  f"former):", file=sys.stderr)
            for (old_folder, new_folder), n in sorted(folder_pair_counts.items()):
                num_left = sum(1 for _ in old_folder.iterdir()) if old_folder.is_dir() else 0
                print(f"    {old_folder.name} -> {new_folder.name}: {n}, {num_left} left",
                      file=sys.stderr)

        if files_in_several_archives:
            print(f"WARN: {len(files_in_several_archives)} image(s) are replaced by more "
                  f"than one patch archive (the last archive in PATCH_ARCHIVE_ORDER wins):",
                  file=sys.stderr)
            _print_path_examples(sorted(files_in_several_archives), out)

        print(f"  {replaced + added} file(s) merged: {replaced} replaced the file at "
              f"their path, {added} have a new path.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="iRAP-Vietnam dataset root.")
    parser.add_argument("--ignore-duplicates", action="store_true",
                        help="Proceed when archives share post-strip paths, and "
                             "keep one copy of each.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt before wiping a "
                             "non-empty images directory.")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Parallel extractions. Default: "
                             "min(num_archives, 4).")
    parser.add_argument("--patches-only", action="store_true",
                        help="Apply only the patch archives, to the images that an "
                             "earlier run extracted from the regular archives.")
    args = parser.parse_args(argv)

    rars = layout.rars_dir(args.data_dir)
    out = layout.images_dir(args.data_dir)
    all_archives = sorted(rars.glob("*.rar")) if rars.is_dir() else []
    if not all_archives:
        print(f"No .rar archives found in {rars}", file=sys.stderr)
        return 1

    try:
        regular_archives, patch_archives = split_regular_and_patch_archives(all_archives)
    except ValueError as e:
        print(f"{e} The archives are in {rars}.", file=sys.stderr)
        return 1

    binary, kind = find_extractor()
    print(f"Using extractor: {binary} ({kind})")
    print(f"Found {len(all_archives)} archive(s):")
    for kind_name, archives in (("Regular", regular_archives), ("Patch", patch_archives)):
        print(f"  {kind_name} archives ({len(archives)}):")
        for a in archives:
            print(f"    {a.name}  ({a.stat().st_size / 1e9:.2f} GB)")

    if args.patches_only:
        if not (out.is_dir() and any(out.iterdir())):
            print(f"--patches-only needs the extracted images, but {out} is empty or "
                  f"missing.", file=sys.stderr)
            return 1
        print("Skipping the regular archives (--patches-only).")
        regular_archives = []

    if regular_archives:
        print(f"Indexing {len(regular_archives)} regular archive(s)...")
    archive_entries = _index_archives(binary, kind, regular_archives)

    dup_out = layout.images_duplicates_dir(args.data_dir)

    if archive_entries:
        print("Checking for duplicate basenames across regular archives...")
        by_name, collisions, total = _compute_collisions(archive_entries)
        print(f"  {len(by_name)} unique basenames, {total / 1e9:.2f} GB total "
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

        _extract_parallel(binary, kind, regular_archives, archive_entries,
                          lambda _a: out, jobs=args.jobs, label="regular")

        print("Flattening splitN/ wrappers...")
        _flatten_split_wrappers(out)

    _apply_patch_archives(binary, kind, patch_archives, out,
                          layout.patch_archives_tmp_dir(args.data_dir), jobs=args.jobs)

    print(f"\nDone. {sum(1 for p in out.rglob('*') if p.is_file())} file(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
