# AURA (Acoustic-Ultrasonic Resonance Analysis)

A complete Python-based multimodal motor health monitoring system. This project uses a Raspberry Pi (Edge Device) to capture real-time multimodal sensor data and a Laptop (Fog Device) to perform high-level AI inference and Visualization.

## System Architecture

*   **Edge Device:** Raspberry Pi 3B+ (Data Collection & Relay Control)
*   **Fog Device:** Laptop (AI Inference, Fusion Engine, & Dashboard)
*   **Communication:** Local WiFi network via TCP Sockets

## Hardware Setup (Raspberry Pi)

### Wiring Diagram Description

1.  **MPU6050 Accelerometer (I2C):**
    *   `SDA` to Pi Pin 3 (SDA)
    *   `SCL` to Pi Pin 5 (SCL)
    *   `VCC` to 3.3V
    *   `GND` to GND
    *   *I2C Address: `0x68`*
2.  **USB Microphone:** Plugged into any USB port on the Pi.
3.  **Pi Camera / ESP32-CAM:** Connected via ribbon cable to CSI port (Pi Camera) or via USB.
4.  **16x2 LCD Display (I2C):**
    *   `SDA` to Pin 3
    *   `SCL` to Pin 5
    *   `VCC` to 5V
    *   `GND` to GND
    *   *I2C Address: `0x27`*
5.  **Relay Module:** IN pin connected to GPIO 17 (Pin 11). This safely controls the motor power.
6.  **Red LED (CRITICAL Status):** Anode to GPIO 27 (Pin 13) via 330Ω resistor, Cathode to GND.
7.  **Green LED (HEALTHY Status):** Anode to GPIO 22 (Pin 15) via 330Ω resistor, Cathode to GND.

---

## Installation & Setup

### 1. Laptop (Fog Device) Setup

1.  Ensure you have Python 3.9+ installed.
2.  Install all required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  In `main.py` and `fusion_engine.py`, locate the `pi_ip` variable and change it to match your Raspberry Pi's IP address on the local network.

### 2. Raspberry Pi (Edge Device) Setup

1.  Enable I2C and Camera via Raspberry Pi configuration:
    ```bash
    sudo raspi-config
    ```
2.  Install system level dependencies:
    ```bash
    sudo apt-get update
    sudo apt-get install python3-smbus i2c-tools python3-opencv portaudio19-dev
    ```
3.  Install Python dependencies required for the Pi:
    ```bash
    pip3 install smbus2 RPi.GPIO RPLCD PyAudio opencv-python numpy
    ```
4.  In `pi_node.py`, locate the `LAPTOP_IP` variable and change it to match your Laptop's IP address.

---

## How to Run

For the system to bind properly, start the Laptop orchestrator first, followed by the Pi node.

### 1. Start the Laptop Orchestrator
```bash
python main.py
```
This will initialize the AI model, start the dashboard, and open listening sockets on ports 5001, 5002, 5003.

### 2. Start the Raspberry Pi Node
```bash
python3 pi_node.py
```
The Pi will begin capturing audio, video, and vibration data, streaming it concurrently to the laptop. It will also listen on port 5004 for a potential "KILL" command.

---

## Technical Pipeline Descriptions

*   **Audio Pipeline:** Captures raw PCM audio, converting it into 128-band Mel-Spectrogram RGB images using `librosa`.
*   **Vision Pipeline:** Computes Dense Optical Flow between consecutive video frames using `cv2.calcOpticalFlowFarneback`, translating vectors into an HSV color-mapped image.
*   **Vibration Pipeline:** Gathers X,Y,Z acceleration arrays. Performs FFT on a 256-sample rolling window to extract dominant frequency, RMS amplitude, and kurtosis.
*   **CNN Model Architecture:** A dual-input Convolutional Neural Network built with TensorFlow. Branch A processes Audio Mel-Spectrograms, Branch B processes Vision Optical Flow. Both are merged to output prediction probabilities.
*   **Sensor Fusion & Decision Engine:** Weights probabilities from pipelines `(0.4 * audio) + (0.35 * vision) + (0.25 * vibration)`. If the resulting score exceeds `0.90`, it triggers a CRITICAL alert, logs the data, and sends a socket command to kill the motor.
*   **Grad-CAM XAI:** Uses `tf.GradientTape` on the final convolution layer of the vision branch to compute a class activation heatmap. This is overlaid onto the live OpenCV feed to visualize spatial regions the AI interprets as anomalies.
