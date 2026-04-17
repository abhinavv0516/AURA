import threading
import time
import numpy as np
import cv2

# Import project modules
from audio_pipeline import AudioPipeline
from vision_pipeline import VisionPipeline
from vibration_pipeline import VibrationPipeline
from temperature_pipeline import TemperaturePipeline
from model import build_model
from fusion_engine import FusionEngine
from gradcam import compute_gradcam, overlay_gradcam
from dashboard import Dashboard

def orchestrator(audio_pipe, vision_pipe, vibe_pipe, temp_pipe, model, fusion, dashboard):
    """
    Main loop that orchestrates data collection, AI inference, and UI updating.
    """
    print("AURA Orchestrator started. Waiting for sensor connections...")
    frame_counter = 0
    
    while True:
        try:
            # 1. Gather latest data from pipeline locks
            with audio_pipe.lock:
                mel_img = audio_pipe.mel_image.copy()
                waveform = audio_pipe.waveform.copy()
                
            with vision_pipe.lock:
                opt_flow = vision_pipe.opt_flow_image.copy()
                current_frame = vision_pipe.current_frame.copy()
                
            with vibe_pipe.lock:
                fft_data = vibe_pipe.fft_data.copy()

            with temp_pipe.lock:
                current_temp = temp_pipe.current_temp
                current_humidity = temp_pipe.current_humidity
                hardware_vib = getattr(temp_pipe, 'current_vib', 0.0)

            # --- UDP Hardware Override for Vibration ---
            if hardware_vib > 0 or getattr(orchestrator, 'using_hardware_vib', False):
                orchestrator.using_hardware_vib = True
                # The user correctly identified that faking an FFT/waveform from a single scalar is senseless.
                # We will pass the true scalar value directly to the dashboard.
                # For the CNN, we provide a basic zero-array so it doesn't crash (shape=128).
                fft_data = np.zeros(128)

            # 2. Expand dims for batch size = 1
            mel_input = np.expand_dims(mel_img, axis=0)
            opt_input = np.expand_dims(opt_flow, axis=0)
            vibe_input = np.expand_dims(fft_data, axis=0)
            temp_input = np.array([[current_temp]])
            
            # 3. Predict via 4-Input CNN
            cnn_prob_arr = model.predict([mel_input, opt_input, vibe_input, temp_input], verbose=0)
            cnn_fused_score = float(cnn_prob_arr[0][0])
            
            # 4. Sensor Fusion & Decision
            final_score, status = fusion.evaluate(cnn_fused_score)
            
            # 5. Grad-CAM XAI (expensive - run every 5 cycles only)
            frame_counter += 1
            if current_frame is not None and current_frame.shape[0] > 0:
                if frame_counter % 5 == 0:
                    heatmap = compute_gradcam(model, [mel_input, opt_input, vibe_input, temp_input], layer_name='vision_last_conv')
                    xai_frame = overlay_gradcam(current_frame, heatmap)
                else:
                    xai_frame = current_frame  # Show raw feed on off-cycles
                    
                try:
                    dashboard.update_video(xai_frame)
                except Exception:
                    pass

            # 6. Update Dashboard Graphs
            try:
                # Compute frequency spectrum for audio
                audio_fft = np.abs(np.fft.rfft(waveform))
                audio_fft_ui = audio_fft[::50] # Downsample for smoother UI rendering
                
                dashboard.update_graphs(audio_fft_ui, fft_data, current_temp, final_score, current_humidity, hardware_vib)
                dashboard.update_status(status)
                
                if status != "HEALTHY":
                    # Simple throttle to avoid spamming the log
                    if not hasattr(orchestrator, 'last_status') or orchestrator.last_status != status:
                        dashboard.log_event(f"State Shift: {status} (Score: {final_score:.2f})")
                        orchestrator.last_status = status
            except Exception:
                pass
                
            # Sleep briefly to free up CPU
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in orchestrator loop: {e}")
            time.sleep(1)

def main():
    print("="*50)
    print("             AURA SYSTEM ONLINE             ")
    print("="*50)
    
    # Setup configuration (Change this to your Pi's actual IP later!)
    pi_ip = '127.0.0.1' 
    
    # 1. Initialize Pipelines
    audio_pipe = AudioPipeline(port=5001)
    vision_pipe = VisionPipeline(port=5002)
    vibe_pipe = VibrationPipeline(port=5003)
    temp_pipe = TemperaturePipeline(port=5005)
    
    # 2. Initialize Engine & Model
    fusion = FusionEngine(pi_ip=pi_ip, cmd_port=5004)
    model = build_model()
    import os
    if os.path.exists('aura_weights.weights.h5'):
        model.load_weights('aura_weights.weights.h5')
        print(">>> Loaded Trained AI Weights! <<<")
    else:
        print("WARNING: Using randomized weights (Not Trained!)")
    
    # 3. Start receiving threads simultaneously
    audio_pipe.start()
    vision_pipe.start()
    vibe_pipe.start()
    temp_pipe.start()
    
    # 4. Initialize Dashboard & Orchestrator
    try:
        dashboard = Dashboard(fusion)
        
        # Start orchestrator in background thread
        orch_thread = threading.Thread(target=orchestrator, 
                                       args=(audio_pipe, vision_pipe, vibe_pipe, temp_pipe, model, fusion, dashboard), 
                                       daemon=True)
        orch_thread.start()
        
        # Start Dashboard UI loop (blocks main thread)
        dashboard.start()
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down gracefully...")
    finally:
        cv2.destroyAllWindows()
        print("AURA SYSTEM OFFLINE.")

if __name__ == "__main__":
    main()
