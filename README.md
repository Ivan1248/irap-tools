# IRAP-Vietnam 360

Tools and pipelines built for the IRAP Vietnam road-survey work. Three independent Python packages:

| Package | Purpose |
|---|---|
| [`irap_video_cutting`](packages/irap_video_cutting/) | Cut MP4 + GPX / WebGIS sidecar files at manual timestamps (CLI + GUI). |
| [`irap_vietnam_360`](packages/irap_vietnam_360/) | Extract perspective images from Insta360 fisheye video using GPS tracks. **Note:** Superseded by [modified Gyroflow](https://github.com/Ivan1248/gyroflow). |
| [`irap_vietnam_data_preparation`](packages/irap_vietnam_data_preparation/) | End-to-end pipeline that turns raw IRAP-Vietnam data into a dataset compatible with the IRAP-BH dataset. |

Each package installs independently:

```bash
uv pip install -e packages/irap_video_cutting
```

Python 3.10+ required. Some packages additionally need `ffmpeg`/`ffprobe` or `unrar`/`7z` on `PATH` — see the package READMEs.
