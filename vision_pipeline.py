import socket
import threading
import struct
import cv2
import numpy as np

PHONE_STREAM_URL = "http://10.22.198.170:8080/video"  # HTTP only - OpenCV does not support HTTPS

class VisionPipeline:
    def __init__(self, port=5002):
        """
        Receives video from IP Webcam (phone) or falls back to simulator socket.
        Computes Dense Optical Flow.
        """
        self.port = port
        self.lock = threading.Lock()
        self.current_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.opt_flow_image = np.zeros((224, 224, 3), dtype=np.float32)
        self.prev_gray = None

    def start(self):
        """Starts phone camera thread AND simulator socket thread in parallel."""
        threading.Thread(target=self._read_phone_camera, daemon=True).start()
        threading.Thread(target=self._receive_from_socket, daemon=True).start()

    def _read_phone_camera(self):
        """Reads MJPEG stream from phone over HTTPS using requests (ignores SSL cert)."""
        import requests
        import time
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Try both HTTPS and HTTP
        urls_to_try = [
            "https://10.22.198.170:8080/video",
            "http://10.22.198.170:8080/video",
        ]
        
        for stream_url in urls_to_try:
            print(f"Trying: {stream_url}")
            try:
                response = requests.get(stream_url, stream=True, verify=False, timeout=5)
                print(f">>> Phone Camera Connected via {stream_url}! <<<")
                frame_count = 0
                bytes_buf = b''
                for chunk in response.iter_content(chunk_size=4096):
                    bytes_buf += chunk
                    # Find JPEG boundaries in the stream
                    start = bytes_buf.find(b'\xff\xd8')
                    end   = bytes_buf.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg_data = bytes_buf[start:end+2]
                        bytes_buf = bytes_buf[end+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.resize(frame, (320, 240))
                            with self.lock:
                                self.current_frame = frame.copy()
                            frame_count += 1
                            if frame_count % 3 == 0:
                                self._process_frame(frame)
                return  # Success - don't try other URLs
            except Exception as e:
                print(f"Failed {stream_url}: {e}")
                continue
        
        print("Phone camera not reachable on any URL.")

    def _receive_from_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        print(f"VisionPipeline fallback: listening on port {self.port}...")
        
        while True:
            conn, addr = server_socket.accept()
            print("Simulator Video Stream connected.")
            data_buffer = b''
            payload_size = struct.calcsize("L")
            
            try:
                while True:
                    while len(data_buffer) < payload_size:
                        data_buffer += conn.recv(4096)
                        
                    packed_msg_size = data_buffer[:payload_size]
                    data_buffer = data_buffer[payload_size:]
                    msg_size = struct.unpack("L", packed_msg_size)[0]
                    
                    while len(data_buffer) < msg_size:
                        data_buffer += conn.recv(4096)
                        
                    frame_data = data_buffer[:msg_size]
                    data_buffer = data_buffer[msg_size:]
                    
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.current_frame = frame.copy()
                        self._process_frame(frame)
                        
            except Exception as e:
                print(f"Vision Pipeline Exception: {e}")
            finally:
                conn.close()

    def _process_frame(self, frame):
        """
        Computes Dense Optical Flow between consecutive frames
        and converts it into an HSV color-mapped image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return
            
        # Dense Optical Flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv = np.zeros((224, 224, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Normalize and store the optical flow image for CNN prediction
        rgb_flow_norm = rgb_flow.astype(np.float32) / 255.0
        
        with self.lock:
            self.opt_flow_image = rgb_flow_norm
            
        self.prev_gray = gray
