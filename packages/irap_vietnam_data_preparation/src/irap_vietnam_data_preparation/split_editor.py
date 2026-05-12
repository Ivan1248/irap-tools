"""Interactive map GUI for manually assigning sections to train/val/test splits.

Requires Streamlit >= 1.35 (for on_select="rerun").

Launch:
    streamlit run irap_vietnam_data_preparation/split_editor.py -- <data_dir>

Reads ``<data_dir>/{segment_id_to_road_data,road_id_to_segment_id_sequence}.json``;
writes ``<data_dir>/splits.json``.

Interaction:
    1. Pick an active split in the sidebar (Train / Val / Test / None).
    2. Draw a rectangle on the map — all sections whose centroid falls
       inside are assigned to the active split.
    3. Undo reverts the last assignment batch; Reset clears all.
    4. Save writes splits.json in the same format as make_splits.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import layout


# ---------------------------------------------------------------------------
# Colours (one per split label)
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test", "none")
COLORS: dict[str, str] = {
    "train": "#4477AA",
    "val":   "#EE6677",
    "test":  "#228833",
    "none":  "#BBBBBB",
}

# ---------------------------------------------------------------------------
# CLI args (streamlit passes everything after "--" as sys.argv)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    args = parser.parse_args()
    args.metadata_dir = layout.metadata_dir(args.data_dir)
    args.output = args.metadata_dir / "splits.json"
    return args


# ---------------------------------------------------------------------------
# Data loading (cached so the large JSON is only read once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_unlabeled_seg_ids(metadata_dir: str) -> list[str]:
    """Return the list of unlabeled seg_ids, or [] if the file does not exist."""
    path = Path(metadata_dir) / layout.UNLABELED_SEG_IDS_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_section_data(
    metadata_dir: str,
) -> tuple[dict, dict]:
    """Return (section_data, seg_to_section).

    section_data[section] = {
        "segs": [seg_id, ...],          # ordered
        "coords": [(lat, lon), ...],    # parallel to segs
        "centroid": (lat, lon),
    }
    seg_to_section[seg_id] = section
    """
    base = Path(metadata_dir)
    with open(base / "segment_id_to_road_data.json", encoding="utf-8") as f:
        seg_to_road: dict = json.load(f)
    with open(base / "road_id_to_segment_id_sequence.json", encoding="utf-8") as f:
        road_to_segs: dict = json.load(f)

    seg_to_section = {sid: data["section"] for sid, data in seg_to_road.items()}

    section_data: dict = {}
    for section, segs in road_to_segs.items():
        coords = [
            (seg_to_road[sid]["lat"], seg_to_road[sid]["lon"])
            for sid in segs
            if sid in seg_to_road
        ]
        if not coords:
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        section_data[section] = {
            "segs": [s for s in segs if s in seg_to_road],
            "coords": coords,
            "centroid": (float(np.median(lats)), float(np.median(lons))),
        }

    return section_data, seg_to_section


def _load_existing_splits(
    output_path: Path,
    seg_to_section: dict,
) -> dict[str, str]:
    """Return section → split from an existing splits.json, or {}."""
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as f:
        raw: dict = json.load(f)
    result: dict[str, str] = {}
    for split, seg_ids in raw.items():
        if split not in SPLITS:
            continue
        for sid in seg_ids:
            sec = seg_to_section.get(str(sid))
            if sec:
                result[sec] = split
    return result


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def _data_bbox(section_data: dict) -> tuple[float, float, float, float] | None:
    """Return (lat_min, lat_max, lon_min, lon_max) over all coords, or None."""
    lats: list[float] = []
    lons: list[float] = []
    for s in section_data.values():
        for lat, lon in s["coords"]:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        return None
    return min(lats), max(lats), min(lons), max(lons)


def _build_figure(
    section_data: dict,
    section_to_split: dict[str, str],
) -> tuple[go.Figure, dict]:
    fig = go.Figure()

    # Layer 1: road polylines — 4 aggregated traces (one per split)
    for split, color in COLORS.items():
        lats: list = []
        lons: list = []
        for sec, sp in section_to_split.items():
            if sp != split:
                continue
            coords = section_data[sec]["coords"]
            lats += [c[0] for c in coords] + [None]
            lons += [c[1] for c in coords] + [None]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=color, width=2),
            name=split,
            hoverinfo="skip",
            showlegend=False,
        ))

    # Layer 2: centroids — single trace, used for box selection
    sections = list(section_data)
    centroid_lats = [section_data[s]["centroid"][0] for s in sections]
    centroid_lons = [section_data[s]["centroid"][1] for s in sections]
    centroid_colors = [COLORS[section_to_split[s]] for s in sections]
    hover_texts = [
        f"{s}<br>{len(section_data[s]['segs'])} segs · {section_to_split[s]}"
        for s in sections
    ]
    fig.add_trace(go.Scattergeo(
        lat=centroid_lats,
        lon=centroid_lons,
        mode="markers",
        marker=dict(size=8, color=centroid_colors),
        customdata=sections,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="centroids",
        showlegend=False,
    ))

    bbox = _data_bbox(section_data)

    fig.update_layout(
        geo=dict(
            projection=dict(type="mercator"),
            fitbounds="locations",
            visible=True,
            showcountries=True, countrycolor="rgba(0,0,0,0.5)",
            showcoastlines=True, coastlinecolor="rgba(0,0,0,0.6)",
            showland=True, landcolor="rgb(243,243,243)",
            showocean=True, oceancolor="rgb(220,235,245)",
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
        "n_sections": len(section_data),
        "n_centroids": len(centroid_lats),
        "bbox": bbox,
    }
    return fig, diag


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Split Editor", layout="wide")
    args = _parse_args()

    section_data, seg_to_section = load_section_data(str(args.metadata_dir))
    unlabeled_seg_ids: list[str] = load_unlabeled_seg_ids(str(args.metadata_dir))

    # ------------------------------------------------------------------
    # Session state init
    # ------------------------------------------------------------------
    if "section_to_split" not in st.session_state:
        existing = _load_existing_splits(args.output, seg_to_section)
        st.session_state.section_to_split = {
            s: existing.get(s, "none") for s in section_data
        }
        st.session_state.history: list[dict] = []

    section_to_split: dict[str, str] = st.session_state.section_to_split

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
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
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem;
        }
        .modebar-container {
            top: 80px !important;
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
    total_segs = sum(len(section_data[s]["segs"]) for s in section_data)
    for split in SPLITS:
        n = sum(
            len(section_data[s]["segs"])
            for s, sp in section_to_split.items()
            if sp == split
        )
        pct = f" ({100 * n / total_segs:.1f}%)" if total_segs and split != "none" else ""
        secs = sum(1 for sp in section_to_split.values() if sp == split)
        st.sidebar.metric(
            split,
            f"{n} segments {pct}, {secs} sections",
            delta_color="off",
        )
    if unlabeled_seg_ids:
        st.sidebar.metric(
            "unlabeled",
            f"{len(unlabeled_seg_ids)} segments",
            delta_color="off",
        )

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    undo_clicked = col1.button("Undo", width='stretch',
                               disabled=not st.session_state.history)
    reset_clicked = col2.button("Reset", width='stretch')

    st.sidebar.markdown("")
    save_clicked = st.sidebar.button("Save splits.json", width='stretch',
                                     type="primary")

    # ------------------------------------------------------------------
    # Handle sidebar actions before rendering the map
    # ------------------------------------------------------------------
    if undo_clicked and st.session_state.history:
        st.session_state.section_to_split = st.session_state.history.pop()
        st.rerun()

    if reset_clicked:
        st.session_state.history.append(dict(section_to_split))
        st.session_state.section_to_split = {s: "none" for s in section_data}
        st.rerun()

    if save_clicked:
        out: dict[str, list] = {s: [] for s in SPLITS if s != "none"}
        for sec, sp in section_to_split.items():
            if sp in out:
                out[sp].extend(section_data[sec]["segs"])
        if unlabeled_seg_ids:
            out["unlabeled"] = unlabeled_seg_ids
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        st.sidebar.success(f"Saved {args.output}")

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    fig, diag = _build_figure(section_data, section_to_split)
    event = st.plotly_chart(
        fig, on_select="rerun", width='stretch',
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [],
            "showTips": False,
            "responsive": True,
        },
    )
    #st.markdown(
    #    "Draw a rectangle to assign sections to the active split. "
    #    "Use the pan tool to navigate without selecting."
    #)


    # ------------------------------------------------------------------
    # Handle map selection
    # ------------------------------------------------------------------
    if event and event.selection and event.selection.points:
        selected_sections = [
            p["customdata"]
            for p in event.selection.points
            if p.get("customdata") is not None
        ]
        if selected_sections:
            prev = dict(section_to_split)
            changed = False
            for sec in selected_sections:
                if sec in section_to_split and section_to_split[sec] != active_split:
                    section_to_split[sec] = active_split
                    changed = True
            if changed:
                st.session_state.history.append(prev)
            st.rerun()


if __name__ == "__main__":
    main()
