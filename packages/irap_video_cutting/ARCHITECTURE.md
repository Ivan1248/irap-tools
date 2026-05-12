# Architecture and Implementation Details

## `cut_manual` — cut MP4 + GPX

### Processing steps

1. **Get video duration** — `ffprobe` reads the video duration and creation
   time.

2. **Build segments** — place cut boundaries at the requested timestamps plus
   the implicit boundaries at `0` and the end of the video.

3. **Keyframe snap** — use `ffprobe` to locate all I-frames and shift each
   segment start backward to the nearest preceding keyframe.  This ensures
   stream-copy cuts start on a clean frame.

4. **Cut MP4** — `ffmpeg -c copy`.  Stream copy avoids quality loss.  The
   `creation_time` metadata of each segment is set to its actual start time.

5. **Cut GPX** — filter points within `[start, end]`, add interpolated
   boundary points at the exact cut times, and re-zero all timestamps so the
   first point of every segment is at `time = 0`.

### Module overview

| Module | Responsibility |
|--------|----------------|
| `cut_manual.py` | CLI entry point |
| `cut_manual_gui.py` | Tkinter GUI |
| `gpx_io.py` | Parse / write GPX XML |
| `video_meta.py` | `ffprobe` video duration + creation time; GPX time alignment |
| `gpx_cut.py` | Cut GPX tracks with boundary interpolation |
| `pipeline.py` | Orchestrates the cutting workflow |


---

## `cut_webgis` — cut MP4 + JSON + GeoJSON + SQL

### Processing steps

1. **Get video duration** — the timestamp of the last GPS point in the `.json`
   sidecar is used as the video duration (no `ffprobe` needed).

2. **Build segments** — place cut boundaries at the requested timestamps plus
   the implicit boundaries at `0` and the end of the video.

3. **Keyframe snap** — use `ffprobe` to locate all I-frames and shift each
   segment start backward to the nearest preceding keyframe.  This ensures
   stream-copy cuts start on a clean frame.

4. **Cut MP4** — `ffmpeg -c copy -movflags faststart -an`.  Stream copy avoids
   quality loss.  `-movflags faststart` moves the `moov` atom to the front of
   the file so the segment is playable via an HTML `<video>` tag without a full
   download.  `-an` strips the audio track.

5. **Cut JSON** — filter GPS points within `[start, end]`, add interpolated
   boundary points at the exact cut times, and re-zero all timestamps so the
   first point of every segment is at `time = 0`.

6. **Derive GeoJSON** — extract the (lon, lat) pairs from the cut JSON and
   write a new `FeatureCollection / LineString`.

7. **Derive SQL** — project the WGS84 coordinates to EPSG:3857 (Web Mercator),
   update the `filename` field to the segment stem, and rewrite the `INSERT`
   statement.

### Module overview

| Module | Responsibility |
|--------|----------------|
| `cut_webgis.py` | CLI entry point |
| `cut_webgis_gui.py` | Tkinter GUI |
| `webgis_io.py` | Parse / write `.json`, `.geojson`, `.sql`; WGS84→EPSG:3857 projection |
| `webgis_cut.py` | Cut JSON/GeoJSON/SQL sidecars with boundary interpolation |
| `pipeline.py` | Orchestrates the cutting workflow |


---

## Shared modules (`irap_video_cutting/shared/`)

| Module | Responsibility |
|--------|----------------|
| `models.py` | `Segment(start_s, end_s)` dataclass |
| `manual_cuts.py` | Parse text cut list; build `Segment` lists |
| `video_keyframes.py` | `ffprobe` I-frame detection; keyframe snapping |
| `ffmpeg_cut.py` | Stream-copy ffmpeg helper used by both tools |
| `cli_runner.py` | Common CLI runner logic for both tools |
| `cut_gui.py` | Shared GUI components (Tkinter) |
| `stem_index.py` | File discovery and grouping by stem |