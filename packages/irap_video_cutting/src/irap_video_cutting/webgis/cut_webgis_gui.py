"""Tkinter GUI for the WebGIS video cutter."""

import sys
from typing import Dict, List, Optional, Tuple

from irap_video_cutting.shared.cut_gui import CutApp
from irap_video_cutting.shared.models import Segment
from irap_video_cutting.shared.stem_index import build_stem_index
from irap_video_cutting.webgis.pipeline import execute_cut, prepare_cut
from irap_video_cutting.webgis.webgis_io import WebgisTrack


class WebgisCutsApp(CutApp):
    window_title = "WebGIS Video Cutter"

    def _build_stem_index(self, input_dir: str) -> Dict[str, str]:
        return build_stem_index(input_dir, extensions={".mp4"})

    def _get_duration_and_segments(
        self, stem: str, video_path: str, cuts: List[float]
    ) -> Optional[Tuple[float, List[Segment], WebgisTrack]]:
        try:
            return prepare_cut(video_path, cuts)
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda err=e, s=stem: self._append_log(f"Prepare failed for {s!r}: {err}"))
            return None

    def _cut_all(
        self,
        stem: str,
        video_path: str,
        segments: List[Segment],
        snapped_starts: List[Optional[float]],
        out_dir: str,
        ctx: WebgisTrack,
    ) -> bool:
        try:
            execute_cut(stem, video_path, segments, snapped_starts, out_dir, ctx)
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda err=e, s=stem: self._append_log(f"  Cut failed for {s!r}: {err}"))
            return False
        return True


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print("Usage: python -m irap_video_cutting.webgis.cut_webgis_gui")
        return 0
    app = WebgisCutsApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
