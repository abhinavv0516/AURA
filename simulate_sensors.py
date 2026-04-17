import socket
import threading
import struct
import time
import numpy as np
import cv2

LAPTOP_IP = '127.0.0.1' # Connecting locally
AUDIO_PORT = 5001
VIDEO_PORT = 5002
VIBE_PORT = 5003
CMD_PORT = 5004
TEMP_PORT = 5005

SIMULATION_START_TIME = time.time()

def is_faulty():
    # Inject anomaly 15 seconds after the script starts
    return (time.time() - SIMULATION_START_TIME) > 15.0

def connect_with_retry(port, name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows reconnects quickly
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while True:
        try:
            sock.connect((LAPTOP_IP, port))
            print(f"{name} Connected")
            return sock
        except:
            time.sleep(1)

def simulate_temperature():
    while True:
        sock = connect_with_retry(TEMP_PORT, "Simulated Temp")
        temp = 35.0
        try:
            while True:
                if is_faulty():
                    temp += np.random.uniform(0.5, 2.0)
                    if temp > 65.0: temp = 65.0
                    humidity = np.random.uniform(40.0, 55.0)  # Rises slightly with heat
                else:
                    temp = 35.0 + np.random.uniform(-0.5, 0.5)
                    humidity = 55.0 + np.random.uniform(-2.0, 2.0)  # Normal room humidity
                # Send temp + humidity as 2 floats (8 bytes)
                sock.sendall(struct.pack('ff', temp, humidity))
                time.sleep(1.0)
        except:
            pass

def simulate_vibration():
    while True:
        sock = connect_with_retry(VIBE_PORT, "Simulated Vibe")
        try:
            while True:
                if is_faulty():
                    # Massive random vibration (Anomaly)
                    x = np.random.normal(0, 1.0)
                    y = np.random.normal(0, 1.0)
                    z = np.random.normal(1.0, 2.5)
                else:
                    # Normal vibration
                    x = np.random.normal(0, 0.05)
                    y = np.random.normal(0, 0.05)
                    z = np.random.normal(1.0, 0.05)
                    
                sock.sendall(struct.pack('fff', x, y, z))
                time.sleep(1/50.0)
        except:
            pass

def simulate_audio():
    while True:
        sock = connect_with_retry(AUDIO_PORT, "Simulated Audio")
        try:
            while True:
                if is_faulty():
                    # High pitch squeal / loud noise
                    noise = np.random.randint(-32000, 32000, 1024, dtype=np.int16)
                else:
                    # Low hum
                    noise = np.random.randint(-3000, 3000, 1024, dtype=np.int16)
                    
                sock.sendall(noise.tobytes())
                time.sleep(1024/44100.0)
        except:
            pass

def simulate_video():
    healthy_path = r"C:\Users\abhin\.gemini\antigravity\brain\8c6f0723-d759-43c0-a770-3923cc913195\bearing_healthy_1776421832534.png"
    faulty_path  = r"C:\Users\abhin\.gemini\antigravity\brain\8c6f0723-d759-43c0-a770-3923cc913195\bearing_faulty_1776421849650.png"
    
    healthy_img = cv2.imread(healthy_path)
    faulty_img  = cv2.imread(faulty_path)
    
    # Fallback if images not found
    if healthy_img is None: healthy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    if faulty_img is None:  faulty_img  = np.full((240, 320, 3), (0, 0, 100), dtype=np.uint8)
    
    healthy_img = cv2.resize(healthy_img, (320, 240))
    faulty_img  = cv2.resize(faulty_img, (320, 240))
    
    while True:
        sock = connect_with_retry(VIDEO_PORT, "Simulated Video")
        try:
            while True:
                if is_faulty():
                    frame = faulty_img.copy()
                    # Subtle jitter to simulate physical vibration
                    jx = np.random.randint(-4, 4)
                    jy = np.random.randint(-4, 4)
                    M = np.float32([[1, 0, jx], [0, 1, jy]])
                    frame = cv2.warpAffine(frame, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
                    # Overlay a DANGER text stamp
                    cv2.putText(frame, "FAULT DETECTED", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frame = healthy_img.copy()
                    # Microscopic vibration so optical flow works
                    jy = np.random.randint(-1, 2)
                    M = np.float32([[1, 0, 0], [0, 1, jy]])
                    frame = cv2.warpAffine(frame, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
                    cv2.putText(frame, "NOMINAL", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                data = buffer.tobytes()
                sock.sendall(struct.pack("L", len(data)))
                sock.sendall(data)
                time.sleep(1/30.0)
        except:
            pass

def mock_command_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', CMD_PORT))
    sock.listen(1)
    print(f"Mock Listener waiting for commands on port {CMD_PORT}...")
    while True:
        conn, addr = sock.accept()
        try:
            data = conn.recv(1024).decode()
            if data == "KILL":
                print("\n>>> MOCK PI RECEIVED KILL COMMAND! SIMULATED MOTOR SHUTDOWN! <<<\n")
        except:
            pass
        finally:
            conn.close()

if __name__ == '__main__':
    print("Starting Simulated Data Streams (Will inject fault at T=15s)...")
    threading.Thread(target=simulate_temperature, daemon=True).start()
    threading.Thread(target=simulate_vibration, daemon=True).start()
    threading.Thread(target=simulate_audio, daemon=True).start()
    threading.Thread(target=simulate_video, daemon=True).start()
    
    mock_command_listener()
