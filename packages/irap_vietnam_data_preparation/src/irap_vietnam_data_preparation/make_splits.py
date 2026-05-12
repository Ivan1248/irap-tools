"""Assign sections to train/val/test splits.

Per-section assignment (Decision 11): whole sections go into one split, so
splits are trivially geographically non-overlapping.

Usage:
    python irap_vietnam_data_preparation/make_splits.py <data_dir>

Reads ``<data_dir>/segment_id_to_road_data.json``;
overwrites ``<data_dir>/splits.json``.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import layout


def assign_splits(
    sections: list[str],
    section_centroids: dict[str, tuple[float, float]],
    ratios: tuple[float, float, float],
    seed: int,
    k_clusters: int = 100,
) -> dict[str, list[str]]:
    """Cluster sections spatially and partition into train/val/test by ``ratios``.

    Allocation rule: cluster section centroids using K-Means. Then walk the
    shuffled clusters and assign each entire cluster to the split whose current
    size is most under its target proportion. This keeps
    small section counts (e.g. n=4 with 80/10/10) sensible — every split
    receives at least one section if the ratio is nonzero.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0; got {ratios} (sum={sum(ratios)})")
    if any(r < 0 for r in ratios):
        raise ValueError(f"Ratios must be non-negative; got {ratios}")

    rng = random.Random(seed)
    
    n_clusters = min(k_clusters, len(sections))
    if n_clusters > 1:
        coords = np.array([section_centroids[s] for s in sections])
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        clusters = defaultdict(list)
        for section, label in zip(sections, labels):
            clusters[label].append(section)
        cluster_list = list(clusters.values())
    else:
        cluster_list = [[s] for s in sections]
        
    rng.shuffle(cluster_list)

    names = ("train", "val", "test")
    targets = dict(zip(names, ratios))
    out: dict[str, list[str]] = {n: [] for n in names}

    n_total = len(sections)
    assigned_sections = 0
    for cluster in cluster_list:
        assigned_sections += len(cluster)
        # current under-fill = target_count - current_count
        # target_count uses fractional progress so early picks land in big splits.
        progress = assigned_sections / n_total if n_total > 0 else 1.0
        deficits = {
            n: targets[n] * progress * n_total - len(out[n])
            for n in names if targets[n] > 0
        }
        pick = max(deficits, key=lambda n: deficits[n])
        out[pick].extend(cluster)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    args = parser.parse_args(argv)

    ratios = (0.8, 0.1, 0.1)
    k_clusters = 100
    seed = 0
    metadata_dir = layout.metadata_dir(args.data_dir)
    seg_to_road_path = metadata_dir / "segment_id_to_road_data.json"
    if not seg_to_road_path.is_file():
        print(f"ERROR: {seg_to_road_path} not found.", file=sys.stderr)
        return 1
    with open(seg_to_road_path, "r", encoding="utf-8") as f:
        seg_to_road = json.load(f)

    section_to_segs: dict[str, list[str]] = defaultdict(list)
    section_lats: dict[str, list[float]] = defaultdict(list)
    section_lons: dict[str, list[float]] = defaultdict(list)
    for sid, data in seg_to_road.items():
        section = data["section"]
        section_to_segs[section].append(sid)
        section_lats[section].append(data["lat"])
        section_lons[section].append(data["lon"])
    
    sections = sorted(section_to_segs)
    section_centroids = {
        s: (float(np.median(section_lats[s])), float(np.median(section_lons[s])))
        for s in sections
    }
    print(f"Found {len(sections)} section(s), {len(seg_to_road)} segment(s).")

    section_splits = assign_splits(
        sections, section_centroids, ratios, seed, k_clusters
    )

    splits_out: dict[str, list[str]] = {n: [] for n in section_splits}
    for split_name, section_list in section_splits.items():
        for s in section_list:
            splits_out[split_name].extend(section_to_segs[s])

    unlabeled_path = layout.unlabeled_seg_ids_path(args.data_dir)
    if unlabeled_path.is_file():
        with open(unlabeled_path, "r", encoding="utf-8") as f:
            splits_out["unlabeled"] = json.load(f)
    else:
        print(f"WARN: {unlabeled_path} not found; 'unlabeled' split omitted.",
              file=sys.stderr)

    for split_name in splits_out:
        n_segs = len(splits_out[split_name])
        if split_name == "unlabeled":
            print(f"  {split_name}: {n_segs} segments")
        else:
            print(f"  {split_name}: {len(section_splits[split_name])} sections, "
                  f"{n_segs} segments")

    out_path = metadata_dir / "splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
