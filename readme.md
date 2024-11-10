# 📹 Multi-Object Tracking Desktop Application

## 📄 Description
This project is a desktop application that leverages **YOLO** (You Only Look Once) for real-time object detection and **Norfair** for object tracking. The application provides a user-friendly **PyQt5** GUI where users can select a device (like a webcam) or upload a video file to analyze object movements in real-time.

## ✨ Features
- **Real-Time Object Detection** using YOLOv5.
- **Multi-Object Tracking** with the Norfair Tracker.
- Supports both **Live Webcam Feed** and **Video File Upload**.
- User-friendly **Graphical Interface** built with PyQt5.

## 🚀 Demo
Here's a quick demonstration of how the application works:

![Demo GIF](demo/demo.gif)

## 🛠️ Technologies Used
- **Python**: Programming Language
- **PyQt5**: GUI Framework
- **YOLOv5**: Object Detection Model
- **Norfair**: Object Tracking Library
- **OpenCV**: Computer Vision Library

## 📦 Requirements
Before running the application, ensure you have the following dependencies installed:

```bash
numpy==1.24.3
opencv-python==4.8.0.76
PyQt5==5.15.9
torch==2.1.0
torchvision==0.16.0
norfair==2.3.6
yolov5==7.0.11


**You can install all required packages using:**

pip install -r requirements.txt

**Project Structure**

📦 multi-object-tracking-app/
├── main.py                # Entry point for the application
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation

**Create a Virtual Environment (Recommended):**

python -m venv venv
source venv/bin/activate   # On macOS/Linux
.\venv\Scripts\activate    # On Windows

**Install Dependencies:**

pip install -r requirements.txt

Run the Application:

python main.py

📄 License

**This project is licensed under the MIT License. See the LICENSE file for more details.**
💬 Contact

**For any questions, issues, or feedback:**

    Your Name: sumirthakur@gmail.com
