"""
Shared Tkinter base class for video-cutting GUIs.

Subclasses must implement three abstract methods:

  _build_stem_index(input_dir)
      Return {stem: full_path} for every video file in input_dir.

  _get_duration_and_segments(stem, video_path, cuts)
      Return (duration_s, segments, ctx) or None to skip this video.
      ctx is opaque data forwarded unchanged to _cut_all.

  _cut_all(stem, video_path, segments, snapped_starts, out_dir, ctx)
      Cut the video and any sidecar files.  Return True on success,
      False to skip the "Done" log entry (error already logged).
"""

import os
import shutil
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Tuple

from irap_video_cutting.shared.manual_cuts import parse_pasted_table
from irap_video_cutting.shared.models import Segment
from irap_video_cutting.shared.stem_index import resolve_stems
from irap_video_cutting.shared.video_keyframes import snap_segments_to_keyframes


class CutApp(tk.Tk):
    window_title = "Video Cutter"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.window_title)

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.clear_output_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        dir_frame = tk.Frame(top)
        dir_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(dir_frame, text="Input directory:").grid(row=0, column=0, sticky="w")
        tk.Entry(dir_frame, textvariable=self.input_dir_var, width=40).grid(row=0, column=1, sticky="we", padx=(4, 4))
        tk.Button(dir_frame, text="Browse", command=self._choose_input_dir).grid(row=0, column=2, padx=(4, 0))

        tk.Label(dir_frame, text="Output directory:").grid(row=1, column=0, sticky="w")
        tk.Entry(dir_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, sticky="we", padx=(4, 4))
        tk.Button(dir_frame, text="Browse", command=self._choose_output_dir).grid(row=1, column=2, padx=(4, 0))

        tk.Checkbutton(
            dir_frame,
            text="Clear output directory before processing",
            variable=self.clear_output_var,
        ).grid(row=2, column=1, sticky="w", padx=(4, 4), pady=(2, 0))

        dir_frame.columnconfigure(1, weight=1)

        tk.Label(top, text="Pasted cuts (videoStem whitespace mm:ss ...):").pack(anchor="w")
        self.text = scrolledtext.ScrolledText(top, height=12)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        btn_frame = tk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Button(btn_frame, text="Preview", command=self._preview).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Run", command=self._run).pack(side=tk.LEFT, padx=(4, 0))

        self.log = scrolledtext.ScrolledText(top, height=10, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.progress = ttk.Progressbar(top, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X)

    def _choose_input_dir(self) -> None:
        d = filedialog.askdirectory(title="Select input directory")
        if d:
            self.input_dir_var.set(d)

    def _choose_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir_var.set(d)

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _parse_mapping(self) -> Dict[str, List[float]]:
        return parse_pasted_table(self.text.get("1.0", tk.END))

    def _preview(self) -> None:
        try:
            mapping = self._parse_mapping()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Parse error", str(e))
            return

        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

        if not mapping:
            self._append_log("No valid rows found.")
            return

        self._append_log(f"Parsed {len(mapping)} video stems:")
        for stem, cuts in mapping.items():
            cuts_str = ", ".join(f"{c:.1f}s" for c in cuts)
            self._append_log(f"  {stem}: {cuts_str}")

    def _run(self) -> None:
        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        try:
            mapping = self._parse_mapping()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Parse error", str(e))
            return

        if not mapping:
            messagebox.showinfo("Nothing to do", "No valid rows found in pasted text.")
            return

        clear_output = self.clear_output_var.get()
        if clear_output and os.path.isdir(output_dir):
            if not messagebox.askyesno(
                "Confirm clear",
                f"Delete all contents of:\n{output_dir}\n\nThis cannot be undone.",
            ):
                return

        self._append_log("Starting batch cutting…")

        thread = threading.Thread(
            target=self._run_worker,
            args=(input_dir, output_dir, mapping, clear_output),
            daemon=True,
        )
        thread.start()

    # --- abstract methods (subclasses must override) ---

    def _build_stem_index(self, input_dir: str) -> Dict[str, str]:
        raise NotImplementedError

    def _get_duration_and_segments(
        self, stem: str, video_path: str, cuts: List[float]
    ) -> Optional[Tuple[float, List[Segment], Any]]:
        raise NotImplementedError

    def _cut_all(
        self,
        stem: str,
        video_path: str,
        segments: List[Segment],
        snapped_starts: List[Optional[float]],
        out_dir: str,
        ctx: Any,
    ) -> bool:
        raise NotImplementedError

    # --- shared worker ---

    def _run_worker(
        self,
        input_dir: str,
        output_dir: str,
        mapping: Dict[str, List[float]],
        clear_output: bool,
    ) -> None:
        if clear_output and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
            self.after(0, lambda: self._append_log(f"Cleared output directory: {output_dir}"))

        try:
            stem_index = self._build_stem_index(input_dir)
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda err=e: self._append_log(f"Error building index: {err}"))
            return

        resolved, unresolved = resolve_stems(stem_index, mapping)
        for prefix in unresolved:
            self.after(0, lambda s=prefix: self._append_log(f"No file found for stem {s!r}."))

        total = len(resolved)
        self.after(0, lambda t=total: self.progress.configure(maximum=t, value=0))

        for idx, (stem, video_path, cuts) in enumerate(resolved, start=1):
            result = self._get_duration_and_segments(stem, video_path, cuts)
            if result is None:
                continue
            duration_s, segments, ctx = result

            seg_summary = ", ".join(f"[{seg.start_s:.1f}s–{seg.end_s:.1f}s]" for seg in segments)
            self.after(
                0,
                lambda s=stem, d=duration_s, ss=seg_summary: self._append_log(f"{s}: {d:.1f}s → {ss}"),
            )

            snapped_starts: List[Optional[float]] = snap_segments_to_keyframes(video_path, segments)
            if all(s is not None for s in snapped_starts):
                deltas = [seg.start_s - snp for seg, snp in zip(segments, snapped_starts)]  # type: ignore[arg-type]
                if any(d > 0.001 for d in deltas):
                    delta_str = ", ".join(f"{d:.3f}s" for d in deltas)
                    self.after(0, lambda s=stem, ds=delta_str: self._append_log(f"  Keyframe snap deltas: {ds}"))
            else:
                self.after(0, lambda s=stem: self._append_log(f"  Keyframe detection failed for {s!r}. Falling back."))

            out_dir = os.path.join(output_dir, os.path.dirname(os.path.relpath(video_path, input_dir)))

            if not self._cut_all(stem, video_path, segments, snapped_starts, out_dir, ctx):
                continue

            self.after(
                0,
                lambda s=stem, v=idx: (self._append_log(f"  Done: {s}"), self.progress.configure(value=v)),
            )

        self.after(0, lambda: self._append_log("Batch cutting completed."))
