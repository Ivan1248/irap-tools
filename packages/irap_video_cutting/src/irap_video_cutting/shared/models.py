from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
