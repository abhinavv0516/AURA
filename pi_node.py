import time
import socket
import threading
import struct
import os
import glob
import glob
import cv2

# ================= Configuration =================
LAPTOP_IP = '10.22.198.25'  # Laptop's IP on the S24 hotspot
AUDIO_PORT = 5001
VIDEO_PORT = 5002
VIBE_PORT  = 5003
CMD_PORT   = 5004
TEMP_PORT  = 5005

# ================= Optional Hardware Imports =================
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    RELAY_PIN  = 17
    RED_LED    = 27
    GREEN_LED  = 22
    GPIO.setup(RELAY_PIN,  GPIO.OUT)
    GPIO.setup(RED_LED,    GPIO.OUT)
    GPIO.setup(GREEN_LED,  GPIO.OUT)
    GPIO.output(RELAY_PIN,  GPIO.LOW)
    GPIO.output(RED_LED,    GPIO.LOW)
    GPIO.output(GREEN_LED,  GPIO.HIGH)
    GPIO_AVAILABLE = True
    print("GPIO: Ready")
except Exception as e:
    GPIO_AVAILABLE = False
    print(f"GPIO not available: {e}")

# ================= DHT11 Setup =================
try:
    import Adafruit_DHT
    DHT_SENSOR = Adafruit_DHT.DHT11
    DHT_PIN = 4  # GPIO pin connected to DHT11 DATA wire (change if needed)
    DHT_AVAILABLE = True
    print("DHT11: Ready on GPIO 4")
except Exception as e:
    DHT_AVAILABLE = False
    print(f"DHT11 not available: {e}")

try:
    import smbus2
    bus = smbus2.SMBus(1)
    MPU6050_ADDR = 0x68
    bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)
    MPU_AVAILABLE = True
    print("MPU6050: Ready")
except Exception as e:
    MPU_AVAILABLE = False
    bus = None
    print(f"MPU6050 not available: {e}")

try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27)
    lcd.write_string('AURA SYSTEM\nONLINE')
    LCD_AVAILABLE = True
    print("LCD: Ready")
except Exception as e:
    LCD_AVAILABLE = False
    lcd = None
    print(f"LCD not available (OK): {e}")

try:
    import pyaudio
    AUDIO_AVAILABLE = True
    print("Audio: Ready")
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"PyAudio not available: {e}")


# ================= Helpers =================
def connect_with_retry(ip, port, name):
    """Connects to the laptop and retries until successful."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            print(f"{name} connected.")
            return sock
        except Exception:
            time.sleep(2)


def read_raw_mpu(addr):
    if not MPU_AVAILABLE: return 0
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low  = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    val  = (high << 8) | low
    return val - 65536 if val > 32768 else val


# ================= DHT11 Temperature =================
def read_dht11():
    """Returns (temperature_c, humidity_pct) from DHT11."""
    if DHT_AVAILABLE:
        try:
            humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
            if temperature is not None and humidity is not None:
                return float(temperature), float(humidity)
        except Exception as e:
            print(f"DHT11 read error: {e}")
    # Fallback
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return float(f.read()) / 1000.0, 0.0
    except Exception:
        return 35.0, 0.0


# ================= Streaming Threads =================
def vibration_stream():
    sock = connect_with_retry(LAPTOP_IP, VIBE_PORT, "Vibration")
    while True:
        try:
            if MPU_AVAILABLE:
                # Use GYROSCOPE registers (better for detecting fan wobble)
                # Gyro X: 0x43, Y: 0x45, Z: 0x47 | sensitivity: 131 LSB/deg/s
                gyro_x = read_raw_mpu(0x43) / 131.0
                gyro_y = read_raw_mpu(0x45) / 131.0
                gyro_z = read_raw_mpu(0x47) / 131.0
            else:
                import random
                gyro_x = random.gauss(0, 2.0)
                gyro_y = random.gauss(0, 2.0)
                gyro_z = random.gauss(0, 2.0)
            sock.sendall(struct.pack('fff', gyro_x, gyro_y, gyro_z))
            time.sleep(1 / 50.0)
        except Exception as e:
            print(f"Vibration error: {e}")
            sock = connect_with_retry(LAPTOP_IP, VIBE_PORT, "Vibration")


def temperature_stream():
    sock = connect_with_retry(LAPTOP_IP, TEMP_PORT, "Temperature")
    while True:
        try:
            temp_c, humidity = read_dht11()
            print(f"Temp: {temp_c:.1f}°C  |  Humidity: {humidity:.1f}%")
            # Send BOTH as 2 floats (8 bytes)
            sock.sendall(struct.pack('ff', temp_c, humidity))
            time.sleep(1.0)
        except Exception as e:
            print(f"Temp error: {e}")
            sock = connect_with_retry(LAPTOP_IP, TEMP_PORT, "Temperature")


def audio_stream():
    if not AUDIO_AVAILABLE:
        print("Audio skipped (no PyAudio).")
        return
    import pyaudio
    sock = connect_with_retry(LAPTOP_IP, AUDIO_PORT, "Audio")
    CHUNK = 1024
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                        input=True, frames_per_buffer=CHUNK)
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                sock.sendall(data)
            except Exception as e:
                print(f"Audio TX error: {e}")
                break
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Audio mic error: {e}")
    p.terminate()
    sock.close()


def video_stream():
    sock = connect_with_retry(LAPTOP_IP, VIDEO_PORT, "Video")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret: break
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            data = buffer.tobytes()
            sock.sendall(struct.pack("L", len(data)))
            sock.sendall(data)
            time.sleep(1 / 15.0)  # 15 FPS to reduce bandwidth
        except Exception as e:
            print(f"Video error: {e}")
            break
    cap.release()
    sock.close()


def command_listener():
    """Listens for KILL command from the laptop."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', CMD_PORT))
    srv.listen(1)
    print(f"Listening for commands on port {CMD_PORT}...")
    while True:
        conn, _ = srv.accept()
        try:
            data = conn.recv(1024).decode()
            if data == "KILL":
                print(">>> KILL COMMAND RECEIVED. STOPPING MOTOR. <<<")
                if GPIO_AVAILABLE:
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                    GPIO.output(RED_LED,   GPIO.HIGH)
                    GPIO.output(GREEN_LED, GPIO.LOW)
                if LCD_AVAILABLE:
                    lcd.clear()
                    lcd.write_string('MOTOR STOPPED\nCRITICAL ALERT')
        except Exception as e:
            print(f"Command error: {e}")
        finally:
            conn.close()


# ================= Main =================
if __name__ == '__main__':
    print("=" * 40)
    print("   AURA Edge Node - Raspberry Pi 3B+   ")
    print("=" * 40)
    print(f"Streaming to laptop at {LAPTOP_IP}")

    threading.Thread(target=temperature_stream, daemon=True).start()
    threading.Thread(target=vibration_stream,   daemon=True).start()
    threading.Thread(target=audio_stream,       daemon=True).start()
    threading.Thread(target=video_stream,       daemon=True).start()

    command_listener()  # Blocks main thread
