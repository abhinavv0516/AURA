# AURA — Multimodal Motor Health Monitor
### Adaptive Ultrasonic & Resonance Analysis (AURA)

AURA is a high-fidelity industrial motor monitoring system that uses multimodal sensor fusion (Vibration, Temperature, Audio, and Computer Vision) to predict and prevent motor failures.

## 🚀 Key Features
- **Japanese Lab Aesthetic**: A clean, "Muji-inspired" light-themed dashboard built with CustomTkinter for a premium professional feel.
- **Multimodal AI Fusion**: Analyzes sensor data using a custom CNN to generate a unified "Fault Probability" score.
- **Real-time Telemetry**: High-precision monitoring of:
  - **Vibration**: ADXL345 High-Stability Time-Series Analyzer.
  - **Temperature**: DHT11 Overheat detection.
  - **Vision**: Grad-CAM (XAI) heatmaps for visual fault localization.
- **Automated Response**: Emergency "Kill Switch" command sent to hardware + instant **WhatsApp Alerts** via Twilio.

## 🛡 Safety Thresholds
- **Vibration**: Triggered at +100 deviation from baseline (High-impact sensitivity).
- **Temperature**: Warning at 32°C | Critical Shutdown at 35°C.

## 🛠 Tech Stack
- **Languages**: Python (Desktop GUI)
- **Frameworks**: CustomTkinter, Matplotlib, TensorFlow/Keras, OpenCV
- **Communication**: UDP Stream for low-latency hardware telemetry
- **Alerts**: Twilio WhatsApp API

## 📋 Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for Twilio (SID, Token, To/From Numbers).
4. Launch the system: `./run_dashboard.ps1`

---
*Developed for AURA Multimodal Motor Health Monitoring Hackathon.*
