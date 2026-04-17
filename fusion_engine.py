import csv
import time
import socket
import os
try:
    from twilio.rest import Client
except ImportError:
    Client = None

class FusionEngine:
    def __init__(self, pi_ip='192.168.1.100', cmd_port=5004):
        """
        Receives the direct fault probability from the Multimodal CNN.
        Logs to CSV and triggers Pi commands.
        """
        self.pi_ip = pi_ip
        self.cmd_port = cmd_port
        self.log_file = 'aura_log.csv'
        self.critical_triggered = False
        
        # Initialize CSV logging file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'CNN_Fused_Score', 'Status'])

        # Twilio WhatsApp Setup (Reads from Environment Variables)
        self.twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
        self.twilio_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
        self.twilio_from = os.environ.get('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886') # Default Twilio Sandbox Number
        self.twilio_to = os.environ.get('TWILIO_WHATSAPP_TO', '') # Your number
        
        self.twilio_client = None
        if Client and self.twilio_sid and self.twilio_token:
            try:
                self.twilio_client = Client(self.twilio_sid, self.twilio_token)
                print("Twilio WhatsApp Integration Ready!")
            except Exception as e:
                print(f"Twilio setup error: {e}")

    def evaluate(self, cnn_fused_score):
        """
        Evaluates the direct CNN output probability.
        """
        status = "HEALTHY"
        if cnn_fused_score > 0.78:   # Critical - triggers Kill Switch
            status = "CRITICAL"
            if not self.critical_triggered:
                self.send_kill_command()
                self.send_whatsapp_alert()
                self.critical_triggered = True
        elif cnn_fused_score > 0.60:  # Warning
            status = "WARNING"
            
        # Log all readings to CSV with timestamp
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 
                             round(cnn_fused_score, 4), 
                             status])
            
        return cnn_fused_score, status

    def send_kill_command(self):
        """Sends KILL command string to Pi via socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.pi_ip, self.cmd_port))
            sock.sendall(b"KILL")
            sock.close()
            print("CRITICAL ALERT: KILL command sent to Pi!")
        except Exception as e:
            print(f"Failed to send KILL command to Pi: {e}")

    def send_whatsapp_alert(self):
        """Sends a WhatsApp alert via Twilio."""
        if not self.twilio_client or not self.twilio_to:
            print("WhatsApp Alert Skipped (Twilio credentials not set in environment).")
            return
            
        try:
            # Twilio requires the 'whatsapp:' prefix
            to_number = self.twilio_to if self.twilio_to.startswith('whatsapp:') else f"whatsapp:{self.twilio_to}"
            
            message = self.twilio_client.messages.create(
                from_=self.twilio_from,
                body="🚨 *AURA CRITICAL ALERT* 🚨\nMotor fault detected! Emergency shutdown sequence initiated. Please inspect the unit immediately.",
                to=to_number
            )
            print(f"WhatsApp Alert Sent Successfully! (SID: {message.sid})")
        except Exception as e:
            print(f"Failed to send WhatsApp alert: {e}")
