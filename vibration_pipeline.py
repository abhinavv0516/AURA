import socket
import threading
import struct
import numpy as np
from scipy.stats import kurtosis

class VibrationPipeline:
    def __init__(self, port=5003):
        """
        Receives MPU6050 X, Y, Z data stream from Pi via socket.
        """
        self.port = port
        self.lock = threading.Lock()
        self.window = []
        self.fft_data = np.zeros(128)

    def start(self):
        """Starts the vibration receiving thread."""
        threading.Thread(target=self._receive_data, daemon=True).start()

    def _receive_data(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        print(f"VibrationPipeline listening on port {self.port}...")
        
        while True:
            conn, addr = server_socket.accept()
            print("Vibration Stream connected.")
            try:
                while True:
                    data = conn.recv(12) # 3 floats * 4 bytes
                    if not data or len(data) < 12:
                        break
                    
                    acc_x, acc_y, acc_z = struct.unpack('fff', data)
                    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    
                    with self.lock:
                        self.window.append(magnitude)
                        # Keep a rolling window of 256 samples
                        if len(self.window) > 256:
                            self.window.pop(0)
                            
                        if len(self.window) == 256:
                            self._analyze()
            except Exception as e:
                print(f"Vibration Pipeline Exception: {e}")
            finally:
                conn.close()

    def _analyze(self):
        """
        Computes FFT to be consumed by the deep learning model.
        """
        data = np.array(self.window)
        
        # Compute FFT
        fft_vals = np.abs(np.fft.rfft(data))
        
        # Save to be pulled by orchestrator (128 bins exactly)
        self.fft_data = fft_vals[:128].copy()

