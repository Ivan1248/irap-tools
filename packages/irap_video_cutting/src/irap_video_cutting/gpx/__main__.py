"""Run the GPX video cutter.

With no arguments, launches the GUI.
With arguments, runs the CLI.
"""

import sys

from irap_video_cutting.gpx import cut_manual


def main() -> int:
    argv = sys.argv[1:]

    # No arguments: launch GUI
    if not argv:
        from irap_video_cutting.gpx import cut_manual_gui

        return cut_manual_gui.main(argv)

    # With arguments: run CLI (including --help)
    return cut_manual.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
