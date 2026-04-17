import socket
import threading
import csv
import time
import os
import struct

class TemperaturePipeline:
    def __init__(self, port=5005):
        """
        Receives UDP data stream (vib,temp,hum) from custom Pi script.
        """
        self.port = port
        self.lock = threading.Lock()
        
        self.current_temp     = 35.0   # °C baseline
        self.current_humidity = 0.0    # % RH baseline
        self.current_vib      = 0.0    # raw scalar from Pi
        
        # Setup CSV logger for real hardware training data
        self.dataset_file = 'hardware_training_dataset.csv'
        if not os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Vibration_Raw', 'Temperature_C', 'Humidity_RH'])

    def start(self):
        """Starts the UDP receiving thread."""
        threading.Thread(target=self._receive_data, daemon=True).start()

    def _receive_data(self):
        # Use UDP (SOCK_DGRAM) to match friend's receiver.py
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.port))
        print(f"Hardware UDP Pipeline listening on port {self.port}...")
        
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                if not data: continue
                
                msg = data.decode().strip()
                # Parse "237,27.6,41"
                parts = msg.split(",")
                if len(parts) >= 3:
                    vib_val = float(parts[0])
                    temp_c  = float(parts[1])
                    hum_val = float(parts[2])
                    
                    with self.lock:
                        self.current_vib      = vib_val
                        self.current_temp     = temp_c
                        self.current_humidity = hum_val
                        
                    # Log to CSV to prove data collection to judges
                    with open(self.dataset_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), vib_val, temp_c, hum_val])
                        
            except Exception as e:
                # Silently catch decode/split errors to avoid spam
                pass
