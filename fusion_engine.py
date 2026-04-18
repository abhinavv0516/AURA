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
        self.last_alert_time = 0
        self.alert_cooldown = 300 # Send WhatsApp alert at most once every 5 minutes
        
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

    def evaluate(self, cnn_fused_score, current_temp=0.0, hardware_vib=0.0):
        """
        Evaluates system health based on AI score, temperature, and vibration.
        """
        status = "HEALTHY"
        
        # Rule 1: AI Score Over Critical
        if cnn_fused_score > 0.78:
            status = "CRITICAL"
        # Rule 2: Overheat (User requested 35C critical)
        elif current_temp >= 35.0:
            status = "CRITICAL"
        # Rule 3: Extreme Vibration (User requested high threshold: +100 above baseline)
        elif hardware_vib >= 337.0:
            status = "CRITICAL"
        # Rule 4: Warnings (+50 above baseline)
        elif cnn_fused_score > 0.60 or current_temp >= 32.0 or hardware_vib >= 287.0:
            status = "WARNING"
            
        if status == "CRITICAL":
            if not self.critical_triggered:
                self.send_kill_command()
                self.send_whatsapp_alert(cnn_fused_score, current_temp, hardware_vib)
                self.critical_triggered = True
        elif status == "HEALTHY":
            # Reset critical trigger if system recovers (manual reset might be safer for demos)
            # self.critical_triggered = False 
            pass
            
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

    def send_whatsapp_alert(self, score, temp, vib):
        """Sends a detailed WhatsApp alert via Twilio with throttling."""
        if not self.twilio_client or not self.twilio_to:
            print("WhatsApp Alert Skipped (Twilio credentials not set).")
            return
            
        # Throttle alerts to avoid spamming
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            print(f"WhatsApp alert throttled (Next alert available in {int(self.alert_cooldown - (now - self.last_alert_time))}s)")
            return
            
        try:
            to_number = self.twilio_to if self.twilio_to.startswith('whatsapp:') else f"whatsapp:{self.twilio_to}"
            
            # Construct a rich diagnostic message
            body = (
                f"🚨 *AURA EMERGENCY SYSTEM* 🚨\n\n"
                f"Critical motor fault detected! Emergency shutdown initiated.\n\n"
                f"*DIAGNOSTICS:*\n"
                f"• AI Fault Score: {score:.2f}\n"
                f"• Temperature: {temp:.1f}°C\n"
                f"• Vibration Index: {vib:.1f}\n\n"
                f"Please inspect the hardware immediately. v2.4.1"
            )
            
            message = self.twilio_client.messages.create(
                from_=self.twilio_from,
                body=body,
                to=to_number
            )
            self.last_alert_time = now
            print(f"WhatsApp Alert Sent! (SID: {message.sid})")
        except Exception as e:
            print(f"Failed to send WhatsApp alert: {e}")
