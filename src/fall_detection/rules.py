from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class BBoxObservation:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def width(self) -> float:
        return max(1e-6, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1e-6, self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


class FallStateMachine:
    def __init__(
        self,
        aspect_ratio_threshold: float,
        center_drop_threshold: float,
        min_fall_frames: int,
        recovery_frames: int,
    ) -> None:
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.center_drop_threshold = center_drop_threshold
        self.min_fall_frames = min_fall_frames
        self.recovery_frames = recovery_frames

        self.history: Deque[BBoxObservation] = deque(maxlen=max(min_fall_frames * 2, 10))
        self.fall_counter = 0
        self.normal_counter = 0
        self.is_fall = False

    def update(self, obs: BBoxObservation, frame_height: int) -> tuple[bool, dict]:
        self.history.append(obs)

        drop_score = 0.0
        if len(self.history) >= 2:
            dy = obs.center_y - self.history[0].center_y
            drop_score = dy / max(float(frame_height), 1.0)

        aspect_hit = obs.aspect_ratio >= self.aspect_ratio_threshold
        drop_hit = drop_score >= self.center_drop_threshold
        current_fall_like = aspect_hit and drop_hit

        if current_fall_like:
            self.fall_counter += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            if self.normal_counter >= self.recovery_frames:
                self.fall_counter = 0

        if self.fall_counter >= self.min_fall_frames:
            self.is_fall = True
        elif self.normal_counter >= self.recovery_frames:
            self.is_fall = False

        info = {
            "aspect_ratio": round(obs.aspect_ratio, 4),
            "drop_score": round(drop_score, 4),
            "fall_counter": self.fall_counter,
            "normal_counter": self.normal_counter,
            "fall_like": current_fall_like,
        }
        return self.is_fall, info
