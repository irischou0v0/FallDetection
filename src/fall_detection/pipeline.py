from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from .config import PipelineConfig
from .rules import BBoxObservation, FallStateMachine


class FallDetectionPipeline:
    def __init__(self, config: PipelineConfig, log_path: str | Path = "logs/events.csv") -> None:
        self.config = config
        self.model = YOLO(config.model_path)
        self.state = FallStateMachine(
            aspect_ratio_threshold=config.rule.aspect_ratio_threshold,
            center_drop_threshold=config.rule.center_drop_threshold,
            min_fall_frames=config.rule.min_fall_frames,
            recovery_frames=config.rule.recovery_frames,
        )
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "conf", "status"])

    def _get_person_box(self, frame) -> Optional[BBoxObservation]:
        results = self.model.predict(
            frame,
            imgsz=self.config.input_size,
            device=self.config.device,
            conf=self.config.rule.confidence_threshold,
            verbose=False,
        )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best = None
        for b in boxes:
            cls_id = int(b.cls.item())
            if cls_id != 0:  # COCO person
                continue
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            obs = BBoxObservation(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf)
            if best is None or obs.conf > best.conf:
                best = obs
        return best

    def process_video(self, source: str, output_path: Optional[str] = None) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            obs = self._get_person_box(frame)
            status = "normal"
            if obs is not None:
                is_fall, info = self.state.update(obs, frame.shape[0])
                status = "fall" if is_fall else "normal"

                color = (0, 0, 255) if is_fall else (0, 255, 0)
                cv2.rectangle(frame, (int(obs.x1), int(obs.y1)), (int(obs.x2), int(obs.y2)), color, 2)
                cv2.putText(
                    frame,
                    f"{status} ar={info['aspect_ratio']} drop={info['drop_score']}",
                    (int(obs.x1), max(25, int(obs.y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([frame_id, obs.x1, obs.y1, obs.x2, obs.y2, obs.conf, status])

            if writer is not None:
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()
