# IRAP tools

Tools and pipelines built for the IRAP Vietnam road-survey work.

| Package | Purpose |
|---|---|
| [`irap_video_cutting`](packages/irap_video_cutting/) | Cut MP4 + GPX / WebGIS sidecar files at manual timestamps (CLI + GUI). |
| [`irap_vietnam_360`](packages/irap_vietnam_360/) | Extract perspective images from Insta360 fisheye video using GPS tracks. **Note:** Superseded by [modified Gyroflow](https://github.com/Ivan1248/gyroflow). |
| [`irap_vietnam_data_preparation`](packages/irap_vietnam_data_preparation/) | End-to-end pipeline that turns raw IRAP-Vietnam data into a dataset compatible with the IRAP-BH dataset. |
| [`irap_data`](packages/irap_data/) | IRAP-BH and IRAP-Vietnam metadata (torch-free), dataset loaders, a dataset viewer (CLI `irap-dataset-viewer`) and a dataset statistics report (CLI `irap-dataset-stats`). |
| [`irap_evaluation`](packages/irap_evaluation/) | Prediction file format, evaluation with bootstrap intervals, ensembling, tables and coding-table export for saved model predictions (CLI `irap-eval`). |
