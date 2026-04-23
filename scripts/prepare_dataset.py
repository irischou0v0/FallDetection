"""将公开视频按论文流程转换为YOLO训练格式：抽帧、划分、生成yaml。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, sample_interval: int = 5) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % sample_interval != 0:
            continue
        out = output_dir / f"{video_path.stem}_{idx:06d}.jpg"
        cv2.imwrite(str(out), frame)
        saved += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, required=True, help="视频目录")
    parser.add_argument("--out", type=str, default="datasets/custom")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    out_root = Path(args.out)
    raw_frames = out_root / "images" / "raw"
    raw_frames.mkdir(parents=True, exist_ok=True)

    total = 0
    for p in sorted(videos_dir.glob("*")):
        if p.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            continue
        total += extract_frames(p, raw_frames, sample_interval=args.interval)

    images = list(raw_frames.glob("*.jpg"))
    random.shuffle(images)
    split = int(len(images) * 0.8)
    train, val = images[:split], images[split:]

    for part_name, items in (("train", train), ("val", val)):
        target = out_root / "images" / part_name
        target.mkdir(parents=True, exist_ok=True)
        for img in items:
            img.rename(target / img.name)

    print(f"完成抽帧: {total} 张")
    print("下一步：请用Label Studio或CVAT对 images/train 和 images/val 进行person框标注。")


if __name__ == "__main__":
    main()
