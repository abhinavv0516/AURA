$env:TWILIO_ACCOUNT_SID="YOUR_TWILIO_SID"
$env:TWILIO_AUTH_TOKEN="YOUR_TWILIO_TOKEN"
$env:TWILIO_WHATSAPP_TO="whatsapp:+91XXXXXXXXXX"  # <--- PUT YOUR NUMBER HERE

Write-Host "Starting AURA Dashboard with WhatsApp Alerts Enabled..." -ForegroundColor Cyan
C:\Users\abhin\OneDrive\Desktop\AURA\.venv\Scripts\python.exe main.py
