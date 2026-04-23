from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fall_detection.config import load_pipeline_config
from fall_detection.pipeline import FallDetectionPipeline


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    error_signal = pyqtSignal(str)

    def __init__(self, source: str, config_path: str) -> None:
        super().__init__()
        self.source = source
        self.config_path = config_path

    def run(self) -> None:
        try:
            cfg = load_pipeline_config(self.config_path)
            pipeline = FallDetectionPipeline(cfg)
            cap = cv2.VideoCapture(self.source)
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_signal.emit(img)
            cap.release()
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLOv8 跌倒检测系统")
        self.resize(1100, 700)

        self.video_label = QLabel("请选择视频")
        self.video_label.setMinimumSize(900, 600)

        self.open_btn = QPushButton("打开视频")
        self.open_btn.clicked.connect(self.select_video)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.open_btn)

        wrapper = QWidget()
        wrapper.setLayout(layout)
        self.setCentralWidget(wrapper)

        self.thread = None

    def select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if not path:
            return
        config_path = str(Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml")
        self.thread = VideoThread(path, config_path)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()

    def update_frame(self, img: QImage) -> None:
        pixmap = QPixmap.fromImage(img).scaled(
            self.video_label.width(),
            self.video_label.height(),
        )
        self.video_label.setPixmap(pixmap)

    def show_error(self, msg: str) -> None:
        QMessageBox.critical(self, "错误", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
