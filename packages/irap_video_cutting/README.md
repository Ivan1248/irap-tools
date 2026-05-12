Video processing tools for road surveys.

**Requirements:** Python 3.10+, `ffmpeg` and `ffprobe` on `PATH`.


## Installation

From the repository root:

```bash
uv pip install -e packages/irap_video_cutting
```

This registers the console-script entry points so they can be run from anywhere.

Alternatively, the tools can be run without installation using `python -m irap_video_cutting.gpx` or `python -m irap_video_cutting.webgis`.


## Cut list format

The tools tools use following text format. One video per line, the filename stem followed by timestamps:

```text
VID_20241210_172830_00_003 08:45
VID_20241210_105715_00_015 00:13 02:06 03:30
```

- **Timestamps:** `ss`, `mm:ss`, or `hh:mm:ss`
- **Output:** N cut points → N+1 segments
- **Warnings:** Lines with missing or invalid timestamps produce warnings and are skipped
- **Directory structure:** Input subdirectories are preserved in output


---

## `irap_video_cutting.gpx` – Cut MP4 + GPX

Cuts video files at manually specified timestamps. If a `.gpx` file exists alongside the video, it's cut as well.

**Inputs:** `.mp4` (+ optional `.gpx`)

### CLI

```bash
python -m irap_video_cutting.gpx \
  --input-dir  /path/to/videos \
  --output-dir /path/to/output \
  --list-file  cuts.txt \
  [--dry-run] [--clear-output]
```

Omit `--list-file` to read the cut list from stdin. After installation, `cut-manual` is also available.

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | *(required)* | Directory tree containing the input files |
| `--output-dir` | *(required)* | Destination directory |
| `--list-file` | stdin | Text file with cut definitions |
| `--dry-run` | off | Print segment boundaries without writing files |
| `--clear-output` | off | Clear the output directory before processing |

### GUI

```bash
cut-manual-gui

# or without installing:
python -m irap_video_cutting.gpx
```

1. Select **Input directory** and **Output directory**.
2. Paste the cut list into the text box.
3. Click **Preview** to verify the parsed stems and cut times.
4. Click **Run** to start processing.
   - Check **Clear output directory before processing** to wipe the output
     folder first (requires confirmation).

---

## `irap_video_cutting.webgis` – Cut MP4 + JSON + GeoJSON + SQL

Cuts WebGIS video sets (video + GPS tracks + database records) at manually specified timestamps. Produces matching segments with all sidecar files.

**Inputs:** Four files per video set with the same stem:
- `.mp4` – video file
- `.json` – GPS track with video-local timestamps
- `.geojson` – GPS track as GeoJSON LineString (WGS84)
- `.sql` – PostgreSQL INSERT statement (EPSG:3857)

**Outputs:** Numbered segments with all sidecar files (timestamps re-zeroed)

### CLI

```bash
python -m irap_video_cutting.webgis \
  --input-dir  /path/to/video-sets \
  --output-dir /path/to/output \
  --list-file  cuts.txt \
  [--dry-run] [--clear-output]
```

Omit `--list-file` to read the cut list from stdin. After installation, `cut-webgis` is also available.

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | *(required)* | Directory tree containing the input files |
| `--output-dir` | *(required)* | Destination directory |
| `--list-file` | stdin | Text file with cut definitions |
| `--dry-run` | off | Print segment boundaries without writing files |
| `--clear-output` | off | Clear the output directory before processing |

### GUI

```bash
cut-webgis-gui

# or without installing:
python -m irap_video_cutting.webgis
```

1. Select **Input directory** and **Output directory**.
2. Paste the cut list into the text box.
3. Click **Preview** to verify the parsed stems and cut times.
4. Click **Run** to start processing.
   - Check **Clear output directory before processing** to wipe the output
     folder first (requires confirmation).

---

## For developers

See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details, module responsibilities, and processing workflows.
