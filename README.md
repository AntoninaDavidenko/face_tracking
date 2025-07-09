# Real-Time Facial Expression Recognition and 3D Avatar Control System

This project provides a real-time facial tracking system that captures facial expressions, eye movements, and eyebrow positions using camera-based computer vision, then transmits these parameters to a virtual avatar via OSC protocol. Designed for seamless Unity integration.

## Technologies

- **Programming Language**: Python 3.11+
- **Tracking Libraries**: MediaPipe FaceMesh, OpenCV
- **Parameter Transmission**: Python-osc
- **Game Engine**: Unity 2022.3.22f1

## Features

- **Automatic calibration** during startup
- **Mouth tracking**: opening detection, "O" shape recognition, corner movement analysis
- **Eye gaze direction** tracking
- **Individual eye blink** detection
- **Eyebrow expression** recognition: raising and frowning
- **Data smoothing** algorithms
- **OSC output** for direct blendshape parameter control

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/2025-TV-z11/Davydenko_AV.git
cd Davydenko_AV
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash 
pip install mediapipe opencv-python python-osc numpy
```

### 4. Run the application
```bash
python face_tracking.py
```

A camera window will open displaying the video feed. **The first 50 frames are used for calibration** - maintain a neutral facial expression during this period.

## Unity Integration

1. Create or open a Unity project and import `VRChat SDK` and `Avatar 3.0 Emulator`. Import your avatar (`.fbx` format recommended) after SDK installation.

2. Add an `Avatar Descriptor` component to your avatar, connecting the parameter menu and animation controller. Create custom animations for emotion display if needed.

3. Add the `Lyuma AV 3 Osc` component to your avatar.

4. Run the scene and click **"Enable Avatar OSC"** in the automatically generated `Av 3 Runtime` script.
