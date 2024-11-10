import sys
import cv2
import torch
import numpy as np
from typing import List, Optional, Union
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import norfair
from norfair import Detection, Tracker


class YOLO:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load("ultralytics/yolov5", model_name, device=device)

    def __call__(self, img: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> torch.tensor:
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        return self.model(img)


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"
) -> List[Detection]:
    norfair_detections = []
    detections_as_xywh = yolo_detections.xywh[0]
    for detection_as_xywh in detections_as_xywh:
        centroid = np.array([detection_as_xywh[0].item(), detection_as_xywh[1].item()])
        scores = np.array([detection_as_xywh[4].item()])
        norfair_detections.append(
            Detection(points=centroid, scores=scores, label=int(detection_as_xywh[-1].item()))
        )
    return norfair_detections


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize YOLO and Norfair
        self.model = YOLO("yolov5s", device=None)
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        
        # UI setup
        self.setWindowTitle("Object Detection and Counting")
        self.setGeometry(100, 100, 800, 600)
        
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(720, 480)
        
        self.btn_load_video = QPushButton("Load Video", self)
        self.btn_load_video.clicked.connect(self.load_video)
        
        self.btn_use_webcam = QPushButton("Use Webcam", self)
        self.btn_use_webcam.clicked.connect(self.use_webcam)
        
        self.count_label = QLabel("Person Count: 0 | Ball Count: 0", self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.btn_load_video)
        layout.addWidget(self.btn_use_webcam)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)

    def use_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return

            # Perform object detection
            results = self.model(frame)
            detections = yolo_detections_to_norfair_detections(results)
            tracked_objects = self.tracker.update(detections=detections)

            # Draw tracking points
            norfair.draw_points(frame, tracked_objects)
            
            # Count specific objects (e.g., person = 0, ball = 32)
            person_count = sum(obj.label == 0 for obj in tracked_objects)
            ball_count = sum(obj.label == 32 for obj in tracked_objects)

            self.count_label.setText(f"Person Count: {person_count} | Ball Count: {ball_count}")

            # Convert frame to display on PyQt5 label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
