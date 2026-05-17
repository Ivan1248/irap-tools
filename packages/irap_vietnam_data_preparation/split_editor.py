"""Interactive map GUI for manually assigning sequences to train/val/test splits.

Requires Streamlit >= 1.35 (for on_select="rerun").

Launch:
    streamlit run irap_vietnam_data_preparation/split_editor.py -- <data_dir>

Reads ``<data_dir>/{segment_id_to_road_data,road_id_to_segment_id_sequence}.json``;
writes ``<data_dir>/splits.json``.

Interaction:
    1. Pick an active split in the sidebar (Train / Val / Test / None).
    2. Draw a rectangle on the map – all sequences whose centroid falls
       inside are assigned to the active split.
    3. Undo reverts the last assignment batch; Reset clears all.
    4. Save writes splits.json in the same format as make_splits.py.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import layout


# Colours (one per split label)

SPLITS = ("train", "val", "test", "none")
COLORS: dict[str, str] = {
    "train": "#4477AA",
    "val": "#EE6677",
    "test": "#228833",
    "none": "#BBBBBB",
}


def _muted(hex_color: str, t: float = 0.5) -> str:
    """Interpolate ``hex_color`` toward grey ``#BBBBBB`` by factor ``t``."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    gr = 0xBB
    r = round(r * (1 - t) + gr * t)
    g = round(g * (1 - t) + gr * t)
    b = round(b * (1 - t) + gr * t)
    return f"#{r:02X}{g:02X}{b:02X}"


UNLABELED_COLORS: dict[str, str] = {s: _muted(c) for s, c in COLORS.items()}


@dataclass
class Layer:
    """A category of sequences (labeled or unlabeled) rendered as one centroid trace."""

    name: str  # "labeled" | "unlabeled"
    data: dict[str, dict]
    colors: dict[str, str]
    marker_size: int
    key_prefix: str  # "" or "unlabeled_"
    state_key: str  # st.session_state key holding sequence -> split


# CLI args (streamlit passes everything after "--" as sys.argv)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="IRAP_Vietnam dataset root.")
    args = parser.parse_args()
    args.metadata_dir = layout.metadata_dir(args.data_dir)
    args.output = args.metadata_dir / "splits.json"
    return args


# Data loading (cached so the large JSON is only read once per session)


@st.cache_data
def load_unlabeled_sequence_data(metadata_dir: str) -> dict[str, dict]:
    """Return ``{sequence_id: {"segs": [seg_id, ...], "centroid": (lat, lon)}}``.

    Returns ``{}`` if the file does not exist.
    """
    path = Path(metadata_dir) / layout.UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        seq: {
            "segs": list(entry["segs"]),
            "centroid": (float(entry["centroid"][0]), float(entry["centroid"][1])),
        }
        for seq, entry in raw.items()
    }


@st.cache_data
def load_unlocated_seg_ids(metadata_dir: str) -> list[str]:
    """Return unlabeled seg_ids with no derivable map location.

    These come from image folders that have no labeled siblings; they are
    auto-assigned to the ``unlabeled_unlocated`` split. Returns ``[]`` if the
    file does not exist.
    """
    path = Path(metadata_dir) / layout.UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


@st.cache_data
def load_sequence_data(
    metadata_dir: str,
) -> tuple[dict, dict]:
    """Return (sequence_data, seg_to_sequence).

    sequence_data[sequence] = {
        "segs": [seg_id, ...],          # ordered
        "coords": [(lat, lon), ...],    # parallel to segs
        "centroid": (lat, lon),
    }
    seg_to_sequence[seg_id] = sequence
    """
    base = Path(metadata_dir)
    with open(base / "segment_id_to_road_data.json", encoding="utf-8") as f:
        seg_to_road: dict = json.load(f)
    with open(base / "road_id_to_segment_id_sequence.json", encoding="utf-8") as f:
        road_to_segs: dict = json.load(f)

    seg_to_sequence = {sid: data["section"] for sid, data in seg_to_road.items()}

    sequence_data: dict = {}
    for seq, segs in road_to_segs.items():
        coords = [(seg_to_road[sid]["lat"], seg_to_road[sid]["lon"]) for sid in segs if sid in seg_to_road]
        if not coords:
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        sequence_data[seq] = {
            "segs": [s for s in segs if s in seg_to_road],
            "coords": coords,
            "centroid": (float(np.median(lats)), float(np.median(lons))),
        }

    return sequence_data, seg_to_sequence


def _load_existing(
    output_path: Path,
    seg_to_sequence: dict[str, str],
    key_prefix: str = "",
) -> dict[str, str]:
    """Return sequence → split from an existing splits.json.

    Only keys starting with ``key_prefix`` are read; the prefix is stripped
    before validation against :data:`SPLITS`. Returns ``{}`` if the file
    does not exist.
    """
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as f:
        raw: dict = json.load(f)
    result: dict[str, str] = {}
    for key, seg_ids in raw.items():
        if not key.startswith(key_prefix):
            continue
        split = key[len(key_prefix) :]
        if not split or split not in SPLITS:
            continue
        # Skip top-level labeled keys when looking for prefixed ones.
        if key_prefix == "" and "_" in key:
            continue
        for sid in seg_ids:
            seq = seg_to_sequence.get(str(sid))
            if seq:
                result[seq] = split
    return result


# Plotly figure


def _data_bbox(sequence_data: dict) -> tuple[float, float, float, float] | None:
    """Return (lat_min, lat_max, lon_min, lon_max) over all coords, or None."""
    lats: list[float] = []
    lons: list[float] = []
    for s in sequence_data.values():
        for lat, lon in s["coords"]:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        return None
    return min(lats), max(lats), min(lons), max(lons)


def _centroid_trace(layer: Layer, assignments: dict[str, str]) -> go.Scattergeo:
    """Build a single Scattergeo trace of all sequence centroids in ``layer``."""
    sequences = list(layer.data)
    lats = [layer.data[s]["centroid"][0] for s in sequences]
    lons = [layer.data[s]["centroid"][1] for s in sequences]
    colors = [layer.colors[assignments[s]] for s in sequences]
    tag = " (unlabeled)" if layer.name == "unlabeled" else ""
    hover = [f"{s}{tag}<br>{len(layer.data[s]['segs'])} segs · {layer.key_prefix}{assignments[s]}" for s in sequences]
    return go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=layer.marker_size, color=colors, symbol="circle"),
        customdata=[[s, layer.name] for s in sequences],
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name=f"{layer.name}_centroids",
        showlegend=False,
    )


def _build_figure(
    labeled: Layer,
    labeled_assignments: dict[str, str],
    unlabeled: Layer,
    unlabeled_assignments: dict[str, str],
) -> tuple[go.Figure, dict]:
    fig = go.Figure()

    # Layer 1: road polylines – 4 aggregated traces (one per split)
    for split, color in COLORS.items():
        lats: list = []
        lons: list = []
        for seq, sp in labeled_assignments.items():
            if sp != split:
                continue
            coords = labeled.data[seq]["coords"]
            lats += [c[0] for c in coords] + [None]
            lons += [c[1] for c in coords] + [None]
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(color=color, width=2),
                name=split,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Layer 2: unlabeled centroids (drawn under labeled so labeled stays on top)
    if unlabeled.data:
        fig.add_trace(_centroid_trace(unlabeled, unlabeled_assignments))

    # Layer 3: labeled centroids – used for box selection
    fig.add_trace(_centroid_trace(labeled, labeled_assignments))

    bbox = _data_bbox(labeled.data)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            projection=dict(type="mercator"),
            fitbounds="locations",
            visible=True,
            showcountries=True,
            countrycolor="rgba(0,0,0,0.5)",
            showcoastlines=True,
            coastlinecolor="rgba(0,0,0,0.6)",
            showland=True,
            landcolor="rgb(243,243,243)",
            showocean=True,
            oceancolor="rgb(220,235,245)",
            showframe=False,
            resolution=50,
        ),
        dragmode="select",
        modebar=dict(
            orientation="v",
            bgcolor="rgba(255,255,255,1)",
            color="rgba(0,0,0,1)",
            activecolor="rgba(20,80,160,1)",
        ),
        height=700,
        uirevision="stable",  # preserve zoom/pan across reruns
    )
    diag = {
        "n_sequences": len(labeled.data),
        "n_centroids": len(labeled.data),
        "bbox": bbox,
    }
    return fig, diag


def _sidebar_row(layer: Layer, assignments: dict[str, str], split: str, total_segs: int, indent: bool) -> str:
    """Return the HTML for one sidebar metric row."""
    color = layer.colors[split]
    n = sum(len(layer.data[s]["segs"]) for s, sp in assignments.items() if sp == split)
    pct = f" ({100 * n / total_segs:.1f}%)" if total_segs and split != "none" else ""
    seqs = sum(1 for sp in assignments.values() if sp == split)
    label = f"{layer.key_prefix}{split}"
    if indent:
        return (
            f'<div style="margin:0 0 6px 0;font-size:0.85rem;color:#555;">'
            f'<span style="background:{color};display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>'
            f"{label}: {n} segments{pct}, {seqs} sequences"
            f"</div>"
        )
    return (
        f'<div style="margin-bottom:2px;">'
        f'<span style="background:{color};display:inline-block;width:14px;height:14px;'
        f'border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
        f"<strong>{label}</strong>: {n} segments{pct}, {seqs} sequences"
        f"</div>"
    )


# App


def main() -> None:
    st.set_page_config(page_title="Split Editor", layout="wide")
    args = _parse_args()

    sequence_data, seg_to_sequence = load_sequence_data(str(args.metadata_dir))
    unlabeled_sequence_data: dict[str, dict] = load_unlabeled_sequence_data(str(args.metadata_dir))
    unlocated_seg_ids: list[str] = load_unlocated_seg_ids(str(args.metadata_dir))

    labeled = Layer(
        name="labeled",
        data=sequence_data,
        colors=COLORS,
        marker_size=8,
        key_prefix="",
        state_key="sequence_to_split",
    )
    unlabeled = Layer(
        name="unlabeled",
        data=unlabeled_sequence_data,
        colors=UNLABELED_COLORS,
        marker_size=6,
        key_prefix="unlabeled_",
        state_key="unlabeled_sequence_to_split",
    )
    layers: tuple[Layer, ...] = (labeled, unlabeled)

    # Session state init
    if labeled.state_key not in st.session_state:
        u_seg_to_sequence = {str(sid): seq for seq, entry in unlabeled_sequence_data.items() for sid in entry["segs"]}
        seg_lookup = {labeled.name: seg_to_sequence, unlabeled.name: u_seg_to_sequence}
        for layer in layers:
            existing = _load_existing(args.output, seg_lookup[layer.name], layer.key_prefix)
            st.session_state[layer.state_key] = {s: existing.get(s, "none") for s in layer.data}
        st.session_state.history: list[dict] = []

    assignments = {layer.name: st.session_state[layer.state_key] for layer in layers}

    # Sidebar
    st.sidebar.title("Split Editor")

    active_split = st.sidebar.radio(
        "Active split (draw a box to assign)",
        SPLITS,
        horizontal=False,
    )

    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebarHeader"] {
            display: none;
        }
        [data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stAppViewContainer"] > .main {
            top: 0;
        }
        [data-testid="stMainBlockContainer"] {
            padding: 0rem;
            max-width: 100%;
            overflow: hidden;
        }
        .modebar-container {
            top: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
            font-size: 0.8rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.0rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    totals = {layer.name: sum(len(layer.data[s]["segs"]) for s in layer.data) for layer in layers}
    for split in SPLITS:
        st.sidebar.markdown(
            _sidebar_row(labeled, assignments[labeled.name], split, totals[labeled.name], indent=False),
            unsafe_allow_html=True,
        )
        if unlabeled.data:
            st.sidebar.markdown(
                _sidebar_row(unlabeled, assignments[unlabeled.name], split, totals[unlabeled.name], indent=True),
                unsafe_allow_html=True,
            )

    if unlocated_seg_ids:
        unlocated_color = UNLABELED_COLORS["none"]
        st.sidebar.markdown(
            f'<div style="margin:0 0 6px 0;font-size:0.85rem;color:#555;">'
            f'<span style="background:{unlocated_color};display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>'
            f"unlabeled_unlocated: {len(unlocated_seg_ids)} segments "
            f"(auto-assigned on save)"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    undo_clicked = col1.button("Undo", width="stretch", disabled=not st.session_state.history)
    reset_clicked = col2.button("Reset", width="stretch")

    st.sidebar.markdown("")
    save_clicked = st.sidebar.button("Save splits.json", width="stretch", type="primary")

    def _snapshot() -> dict:
        return {layer.name: dict(assignments[layer.name]) for layer in layers}

    # Handle sidebar actions before rendering the map
    if undo_clicked and st.session_state.history:
        snapshot = st.session_state.history.pop()
        for layer in layers:
            st.session_state[layer.state_key] = snapshot[layer.name]
        st.rerun()

    if reset_clicked:
        st.session_state.history.append(_snapshot())
        for layer in layers:
            st.session_state[layer.state_key] = {s: "none" for s in layer.data}
        st.rerun()

    if save_clicked:
        out: dict[str, list] = {s: [] for s in SPLITS if s != "none"}
        for layer in layers:
            for seq, sp in assignments[layer.name].items():
                if sp == "none":
                    continue
                key = f"{layer.key_prefix}{sp}"
                out.setdefault(key, []).extend(layer.data[seq]["segs"])
        if unlocated_seg_ids:
            out["unlabeled_unlocated"] = list(unlocated_seg_ids)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        st.sidebar.success(f"Saved {args.output}")

    # Map
    fig, diag = _build_figure(
        labeled,
        assignments[labeled.name],
        unlabeled,
        assignments[unlabeled.name],
    )
    event = st.plotly_chart(
        fig,
        on_select="rerun",
        width="stretch",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [],
            "showTips": False,
            "responsive": True,
        },
    )

    # Handle map selection
    if event and event.selection and event.selection.points:
        prev = _snapshot()
        target_by_kind = {layer.name: assignments[layer.name] for layer in layers}
        changed = False
        for p in event.selection.points:
            cd = p.get("customdata")
            if not cd:
                continue
            seq, kind = cd[0], cd[1]
            target = target_by_kind.get(kind)
            if target is not None and seq in target and target[seq] != active_split:
                target[seq] = active_split
                changed = True
        if changed:
            st.session_state.history.append(prev)
            st.rerun()


if __name__ == "__main__":
    main()
