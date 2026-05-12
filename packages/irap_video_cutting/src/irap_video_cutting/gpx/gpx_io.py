import datetime as _dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

_NS = "http://www.topografix.com/GPX/1/1"
_ns = {"gpx": _NS}


def _tag(name: str) -> str:
    return f"{{{_NS}}}{name}"


@dataclass(frozen=True)
class Track:
    """
    Raw GPX track in absolute time.

    All timestamps are seconds since Unix epoch (UTC).
    """

    times_s: List[float]
    lat_deg: List[float]
    lon_deg: List[float]
    ele_m: List[float]

    def __post_init__(self) -> None:
        n = len(self.times_s)
        if not (len(self.lat_deg) == len(self.lon_deg) == len(self.ele_m) == n):
            raise ValueError("times, lat, lon, ele must have identical lengths")


def _parse_iso_timestamp(text: str) -> float:
    dt = _dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.timestamp()


def parse_gpx(path: str) -> Track:
    """
    Parse a GPX file and return a flattened Track.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    times: List[float] = []
    lat: List[float] = []
    lon: List[float] = []
    ele: List[float] = []

    for trkpt in root.findall(".//gpx:trkpt", _ns):
        time_el = trkpt.find("gpx:time", _ns)
        if time_el is None or not time_el.text:
            continue
        times.append(_parse_iso_timestamp(time_el.text))
        lat.append(float(trkpt.get("lat")))
        lon.append(float(trkpt.get("lon")))
        ele_el = trkpt.find("gpx:ele", _ns)
        ele.append(float(ele_el.text) if ele_el is not None and ele_el.text else 0.0)

    if not times:
        raise ValueError(f"No timed track points found in GPX file: {path}")

    # Sort by time and deduplicate identical timestamps (keep last occurrence)
    deduped = []
    for p in sorted(zip(times, lat, lon, ele)):
        if deduped and p[0] == deduped[-1][0]:
            deduped[-1] = p
        else:
            deduped.append(p)
    times_s, lat_deg, lon_deg, ele_m = map(list, zip(*deduped))

    return Track(times_s=times_s, lat_deg=lat_deg, lon_deg=lon_deg, ele_m=ele_m)


def save_gpx(track: Track, path: str) -> None:
    """
    Save a Track object to a GPX file.
    """
    ET.register_namespace("", _NS)
    root = ET.Element(_tag("gpx"), attrib={"version": "1.1", "creator": "irap_video_cutting"})
    trkseg = ET.SubElement(ET.SubElement(root, _tag("trk")), _tag("trkseg"))

    for t, la, lo, el in zip(track.times_s, track.lat_deg, track.lon_deg, track.ele_m):
        dt = _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
        trkpt = ET.SubElement(trkseg, _tag("trkpt"), attrib={"lat": f"{la:.7f}", "lon": f"{lo:.7f}"})
        ET.SubElement(trkpt, _tag("ele")).text = f"{el:.3f}"
        ET.SubElement(trkpt, _tag("time")).text = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
        f.write("\n")
