import os
from typing import List

import numpy as np

from irap_video_cutting.gpx.gpx_io import Track, parse_gpx, save_gpx
from irap_video_cutting.gpx.video_meta import TimeAlignment
from irap_video_cutting.shared.models import Segment


def cut_gpx_segments(
    gpx_path: str,
    segments: List[Segment],
    alignment: TimeAlignment,
    output_dir: str,
    base_name: str,
    snapped_starts_s: List[float | None] | None = None,
) -> None:
    """
    Cut a GPX file into multiple segments based on video-local timestamps.
    """
    track = parse_gpx(gpx_path)
    os.makedirs(output_dir, exist_ok=True)

    for idx, seg in enumerate(segments):
        actual_start_s = (
            snapped_starts_s[idx] if snapped_starts_s and snapped_starts_s[idx] is not None else seg.start_s
        )
        actual_end_s = seg.end_s

        start_epoch = actual_start_s - alignment.offset_s
        end_epoch = actual_end_s - alignment.offset_s

        interp_lat_start = float(np.interp(start_epoch, track.times_s, track.lat_deg))
        interp_lon_start = float(np.interp(start_epoch, track.times_s, track.lon_deg))
        interp_ele_start = float(np.interp(start_epoch, track.times_s, track.ele_m))
        interp_lat_end = float(np.interp(end_epoch, track.times_s, track.lat_deg))
        interp_lon_end = float(np.interp(end_epoch, track.times_s, track.lon_deg))
        interp_ele_end = float(np.interp(end_epoch, track.times_s, track.ele_m))

        seg_times: List[float] = [start_epoch]
        seg_lat: List[float] = [interp_lat_start]
        seg_lon: List[float] = [interp_lon_start]
        seg_ele: List[float] = [interp_ele_start]

        for t, lat, lon, ele in zip(track.times_s, track.lat_deg, track.lon_deg, track.ele_m):
            if start_epoch < t < end_epoch:
                seg_times.append(t)
                seg_lat.append(lat)
                seg_lon.append(lon)
                seg_ele.append(ele)

        seg_times.append(end_epoch)
        seg_lat.append(interp_lat_end)
        seg_lon.append(interp_lon_end)
        seg_ele.append(interp_ele_end)

        seg_track = Track(times_s=seg_times, lat_deg=seg_lat, lon_deg=seg_lon, ele_m=seg_ele)
        output_path = os.path.join(output_dir, f"{base_name}_{idx + 1}.gpx")
        save_gpx(seg_track, output_path)
