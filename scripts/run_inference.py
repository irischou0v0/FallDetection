from __future__ import annotations

import argparse

from fall_detection.config import load_pipeline_config
from fall_detection.pipeline import FallDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--source", required=True, help="视频路径或摄像头索引（0）")
    parser.add_argument("--output", default="outputs/result.mp4")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    pipe = FallDetectionPipeline(cfg)
    pipe.process_video(args.source, args.output)
    print(f"推理完成，结果输出: {args.output}")


if __name__ == "__main__":
    main()
