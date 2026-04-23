# 基于 YOLOv8 的行人跌倒检测系统（工程版）

本仓库根据《基于Yolov模型的行人跌倒检测系统-初稿》中“系统实现”相关章节落地，提供可复现的代码框架：
- 数据集准备与抽帧
- YOLOv8 人体检测训练
- 基于边界框几何+时序规则的跌倒判定
- 日志记录与结果输出
- PyQt5 可视化界面骨架

---

## 1. 项目结构

```text
FallDetection/
├── configs/
│   ├── fall_data.yaml          # YOLO数据集配置
│   ├── pipeline.yaml           # 推理与跌倒规则阈值
│   └── train_hyp.yaml          # 训练超参数建议
├── scripts/
│   ├── prepare_dataset.py      # 按步骤抽帧+划分训练验证集
│   ├── train_yolov8.py         # YOLOv8训练脚本
│   └── run_inference.py        # 视频推理脚本
├── src/
│   ├── fall_detection/
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   └── rules.py
│   └── ui/
│       └── app.py              # PyQt5界面示例
├── DATASET_LINKS.txt           # 数据集链接（单独文件）
├── requirements.txt            # pip环境配置
└── environment.yml             # conda环境配置
```

---

## 2. 数据集选择（高质量且贴合实验流程）

本项目优先推荐以下公开数据集（已单独写入 `DATASET_LINKS.txt`）：
1. **Le2i Fall Detection Dataset**（跌倒检测常用基准，室内真实动作）
2. **UR Fall Detection Dataset**（泛化验证常用）
3. **Multiple Cameras Fall Dataset**（多机位鲁棒性）

> 论文流程对应：Kaggle/公开数据下载 → 抽帧 → 标注 person 框 → 640×640 统一输入 → 8:2 划分训练/验证 → 训练 YOLOv8 → 规则判定跌倒。

---

## 3. 环境配置

### 3.1 pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows使用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 conda
```bash
conda env create -f environment.yml
conda activate yolo-fall-detection
```

---

## 4. 按步骤运行（对应“系统实现”）

## Step 1：准备原始视频
将下载的视频放到一个目录，例如：
```text
datasets/raw_videos/
```

## Step 2：抽帧并划分训练集/验证集
```bash
python scripts/prepare_dataset.py --videos datasets/raw_videos --out datasets/custom --interval 5
```

输出后需要人工标注：
- `datasets/custom/images/train`
- `datasets/custom/images/val`

将标注结果放到：
- `datasets/custom/labels/train`
- `datasets/custom/labels/val`

标注格式：YOLO txt（仅 `person` 类，类别ID=0）。

## Step 3：训练 YOLOv8 人体检测模型
```bash
python scripts/train_yolov8.py --model yolov8n.pt --data configs/fall_data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0
```

训练结果默认在：
```text
runs/fall_detection/yolov8n_exp/
```

## Step 4：执行跌倒检测推理
将最优权重路径写入 `configs/pipeline.yaml` 的 `model_path`，然后运行：
```bash
python scripts/run_inference.py --config configs/pipeline.yaml --source test_videos/demo.mp4 --output outputs/demo_result.mp4
```

输出：
- 视频结果：`outputs/*.mp4`
- 事件日志：`logs/events.csv`

## Step 5：可视化界面（可选）
```bash
PYTHONPATH=src python src/ui/app.py
```

---

## 5. 跌倒判定逻辑（与论文实现一致的工程化版本）

在检测到 `person` 框后，系统会计算并融合：
- 宽高比变化（`aspect_ratio = w/h`）
- 人体中心点下移幅度（`drop_score`）
- 连续帧触发计数（`min_fall_frames`）

当以下条件持续满足时触发 fall：
1. `aspect_ratio >= aspect_ratio_threshold`
2. `drop_score >= center_drop_threshold`
3. 连续帧计数达到 `min_fall_frames`

相关阈值在 `configs/pipeline.yaml` 可调。

---

## 6. 后续可扩展方向

- 将规则判定替换为时序模型（LSTM/Transformer）
- 增加多目标跟踪（ByteTrack）降低多人场景误判
- 导出 ONNX/TensorRT 提升实时性
- 在 UI 中接入报警弹窗、截图、声音告警

---

## 7. 说明

本仓库是按照论文初稿的系统实现流程进行的工程化落地模板，便于你后续继续迭代实验、补充模型优化章节与性能对比数据。
