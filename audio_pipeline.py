import socket
import threading
import numpy as np
import librosa
import cv2

class AudioPipeline:
    def __init__(self, port=5001):
        """
        Receives raw PCM audio stream from Pi via socket.
        """
        self.port = port
        self.lock = threading.Lock()
        self.waveform = np.zeros(44100) # 1 sec waveform buffer at 44.1kHz
        self.mel_image = np.zeros((224, 224, 3), dtype=np.float32)

    def start(self):
        """Starts the audio receiving thread."""
        threading.Thread(target=self._receive_data, daemon=True).start()

    def _receive_data(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        print(f"AudioPipeline listening on port {self.port}...")
        
        while True:
            conn, addr = server_socket.accept()
            print("Audio Stream connected.")
            audio_buffer = b''
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    audio_buffer += data
                    
                    # Wait until we have ~1 second of 16-bit PCM audio (44100 samples * 2 bytes)
                    if len(audio_buffer) >= 88200:
                        chunk = audio_buffer[:88200]
                        audio_buffer = audio_buffer[88200:]
                        
                        # Convert bytes to normalized float array
                        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        with self.lock:
                            self.waveform = audio_data.copy()
                            
                        # Process using Librosa: Convert to Mel-Spectrogram
                        S = librosa.feature.melspectrogram(y=audio_data, sr=44100, n_mels=128, hop_length=512)
                        S_DB = librosa.power_to_db(S, ref=np.max)
                        
                        # Normalize 0 to 1
                        S_norm = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min() + 1e-8)
                        
                        # Resize to 224x224 for CNN model input
                        S_resized = cv2.resize(S_norm, (224, 224))
                        
                        # Stack to create a 3-channel RGB image
                        S_rgb = np.stack((S_resized,)*3, axis=-1)
                        
                        with self.lock:
                            self.mel_image = S_rgb
            except Exception as e:
                print(f"Audio Pipeline Exception: {e}")
            finally:
                conn.close()
