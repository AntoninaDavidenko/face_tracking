# 🎭 Real-Time Face Tracking System

A sophisticated real-time facial tracking system that captures facial expressions, eye movements, and eyebrow positions using computer vision, then transmits these parameters to virtual avatars via OSC protocol. Perfect for Unity integration and VRChat applications.

## 🌟 Features

### 🔍 Advanced Face Detection
- **Real-time facial expression tracking** using MediaPipe FaceMesh
- **Automatic calibration** during startup (first 50 frames)
- **High-precision landmark detection** with 468 facial points
- **Smooth data processing** with built-in filtering algorithms

### 👁️ Eye Tracking Capabilities
- **Gaze direction tracking** for realistic eye movement
- **Individual eye blink detection** for each eye
- **Eye aspect ratio calculation** for natural blinking animation
- **Pupil tracking** for enhanced realism

### 👄 Mouth Expression Analysis
- **Mouth opening detection** with precise measurements
- **"O" shape recognition** for vowel sounds
- **Corner movement analysis** for smile/frown detection
- **Lip sync potential** for speech animation

### 🤨 Eyebrow Expression Recognition
- **Eyebrow raising detection** for surprise expressions
- **Eyebrow frowning recognition** for concern/anger
- **Smooth transition between expressions**

### 🎮 Unity Integration
- **OSC protocol communication** for real-time data transmission
- **VRChat SDK compatibility** with Avatar 3.0
- **Direct blendshape parameter control**
- **Plug-and-play setup** with minimal configuration

## 🛠 Tech Stack

- **Programming Language**: Python 3.11+
- **Computer Vision**: MediaPipe FaceMesh, OpenCV
- **Communication Protocol**: Python-osc
- **Game Engine**: Unity 2022.3.22f1
- **Platform**: Windows, macOS, Linux

## 📋 Prerequisites

- Python 3.11 or higher
- Webcam or external camera
- Unity 2022.3.22f1 (for avatar integration)
- VRChat SDK (for VRChat applications)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AntoninaDavidenko/face_tracking.git
cd face_tracking
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install mediapipe opencv-python python-osc numpy
```

### 4. Run the Application
```bash
python face_tracking.py
```

## 🎯 How to Use

### Initial Setup
1. **Launch the application** - A camera window will open displaying your video feed
2. **Calibration phase** - Maintain a neutral facial expression during the first 50 frames
3. **Start tracking** - Begin making facial expressions to see real-time detection

### Camera Controls
- The system automatically detects your default camera
- Ensure good lighting for optimal tracking performance
- Position your face clearly within the camera frame

## 🎮 Unity Integration Guide

### Step 1: Unity Project Setup
1. Create or open a Unity project
2. Import the **VRChat SDK** and **Avatar 3.0 Emulator**
3. Import your avatar (`.fbx` format recommended) after SDK installation

### Step 2: Avatar Configuration
1. Add an **Avatar Descriptor** component to your avatar
2. Connect the parameter menu and animation controller
3. Create custom animations for emotion display if needed

### Step 3: OSC Integration
1. Add the **[Lyuma AV 3 OSC](https://github.com/lyuma/Av3Emulator)** component to your avatar
2. Configure OSC parameters to match face tracking output

### Step 4: Runtime Setup
1. Run the Unity scene
2. Click **"Enable Avatar OSC"** in the automatically generated `Av 3 Runtime` script
3. Start the face tracking application
4. Enjoy real-time facial expression transfer!

## 📊 OSC Parameters

The system outputs the following OSC parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `/avatar/parameters/Mouth_Open` | Float | 0.0-1.0 | Mouth opening amount |
| `/avatar/parameters/Mouth_O` | Float | 0.0-1.0 | Mouth opening amount O |
| `/avatar/parameters/Smile_R` | Float | 0.0-1.0 | Smile intensity |
| `/avatar/parameters/Smile_L` | Float | 0.0-1.0 | Smile intensity |
| `/avatar/parameters/Blink_L` | Float | 0.0-1.0 | Left eye blink |
| `/avatar/parameters/Blink_R` | Float | 0.0-1.0 | Right eye blink |
| `/avatar/parameters/Eye_x` | Float | -1.0-1.0 | Horizontal eye movement |
| `/avatar/parameters/Eye_y` | Float | -1.0-1.0 | Vertical eye movement |
| `/avatar/parameters/Brow_raise` | Float | 0.0-1.0 | Eyebrow raise |
| `/avatar/parameters/Brow_frown` | Float | 0.0-1.0 | Eyebrow furrow |

## 🛠 Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Verify camera is not in use by another application
- Try different camera index values

**Poor tracking accuracy:**
- Ensure good lighting conditions
- Position face clearly in camera frame
- Complete calibration with neutral expression

**OSC connection failed:**
- Verify Unity is running and OSC is enabled
- Check IP address and port configuration
- Ensure firewall isn't blocking connections

**Performance issues:**
- Reduce camera resolution in settings
- Close unnecessary applications
- Update graphics drivers

## 🎯 Use Cases

- **VRChat Avatars**: Real-time facial expressions for social VR
- **Live Streaming**: Interactive avatar control for content creators
- **Game Development**: Facial animation for characters
- **Digital Art**: Expression-based art installations
- **Accessibility**: Alternative input methods for users with disabilities

## 🚀 Future Enhancements

- [ ] Multi-face tracking support
- [ ] Custom gesture recognition
- [ ] Improved performance optimization
- [ ] Advanced emotion recognition

## 📝 Requirements

```
mediapipe>=0.10.0
opencv-python>=4.8.0
python-osc>=1.8.0
numpy>=1.24.0
```

## 🔗 Useful Links

- [MediaPipe Documentation](https://mediapipe.dev/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [VRChat SDK Documentation](https://creators.vrchat.com/sdk/)
- [Unity Documentation](https://docs.unity.com/en-us)
- [OSC Protocol Specification](http://opensoundcontrol.org/)

---
