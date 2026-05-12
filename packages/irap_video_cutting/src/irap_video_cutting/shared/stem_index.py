"""Walk a directory tree and build a {stem: full_path} index."""

import os
from typing import Dict, Set


def build_stem_index(
    input_dir: str,
    extensions: Set[str] | None = None,
    video_exts: Set[str] | None = None,
) -> Dict[str, str]:
    """
    Walk *input_dir* and return ``{stem: full_path}`` for discovered files.

    Parameters
    ----------
    extensions:
        If given, only index files whose lowercased extension is in this set
        (e.g. ``{".mp4"}``).  Duplicate stems raise ``RuntimeError``.
    video_exts:
        If given (and *extensions* is ``None``), all files are indexed but
        collisions are resolved by preferring files whose extension is in
        *video_exts* over those that are not.  Two files with the same stem
        that are both video or both non-video still raise ``RuntimeError``.
    """
    index: Dict[str, str] = {}

    for root, _, files in os.walk(input_dir):
        for name in files:
            stem, ext = os.path.splitext(name)
            ext = ext.lower()
            path = os.path.join(root, name)

            if extensions is not None and ext not in extensions:
                continue

            if stem not in index:
                index[stem] = path
                continue

            # --- handle duplicate stem ---
            existing_path = index[stem]

            if os.path.abspath(path) == os.path.abspath(existing_path):
                continue

            if extensions is not None or video_exts is None:
                raise RuntimeError(
                    f"Duplicate stem {stem!r}: {existing_path} vs {path}"
                )

            # Disambiguate: prefer video extension over non-video.
            existing_ext = os.path.splitext(existing_path)[1].lower()
            new_is_video = ext in video_exts
            old_is_video = existing_ext in video_exts

            if new_is_video and not old_is_video:
                index[stem] = path
            elif not new_is_video and old_is_video:
                continue
            else:
                raise RuntimeError(
                    f"Multiple files with stem {stem!r} in {input_dir!r}; "
                    f"cannot disambiguate between {existing_path} and {path}."
                )

    return index


def _lookup_by_prefix(
    stem_index: Dict[str, str],
    prefix: str,
) -> tuple[str, str] | None:
    """
    Look up a video by exact stem or unambiguous prefix.

    Returns ``(matched_stem, path)`` on success, or ``None`` if the prefix
    matches zero or more than one stem.
    """
    # Exact match — fast path.
    if prefix in stem_index:
        return prefix, stem_index[prefix]

    matches = [(s, p) for s, p in stem_index.items() if s.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_stems(
    stem_index: Dict[str, str],
    mapping: Dict[str, list[float]],
) -> tuple[list[tuple[str, str, list[float]]], list[str]]:
    """
    Resolve each key in *mapping* against *stem_index* by unambiguous prefix.

    Returns ``(resolved, unresolved)`` where *resolved* is a list of
    ``(full_stem, video_path, cuts)`` tuples and *unresolved* lists the
    prefixes that matched zero or more than one stem.
    """
    resolved: list[tuple[str, str, list[float]]] = []
    unresolved: list[str] = []
    for prefix, cuts in mapping.items():
        match = _lookup_by_prefix(stem_index, prefix)
        if match is None:
            unresolved.append(prefix)
        else:
            resolved.append((match[0], match[1], cuts))
    return resolved, unresolved
