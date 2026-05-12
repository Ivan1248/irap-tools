"""Run the WebGIS video cutter.

With no arguments, launches the GUI.
With arguments, runs the CLI.
"""

import sys

from irap_video_cutting.webgis import cut_webgis


def main() -> int:
    argv = sys.argv[1:]

    # No arguments: launch GUI
    if not argv:
        from irap_video_cutting.webgis import cut_webgis_gui

        return cut_webgis_gui.main(argv)

    # With arguments: run CLI (including --help)
    return cut_webgis.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
