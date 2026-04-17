import numpy as np
import tensorflow as tf
from model import build_model
import os

def generate_synthetic_data(num_samples=200):
    print(f"Generating {num_samples} synthetic multimodal samples...")
    
    audio_data = []
    vision_data = []
    vibe_data = []
    temp_data = []
    labels = []
    
    for i in range(num_samples):
        # 50% Healthy (0), 50% Faulty (1)
        is_faulty = 1 if i >= num_samples // 2 else 0
        
        if not is_faulty:
            # Healthy Data - toy DC motor room temp
            temp = np.random.normal(38.0, 2.0)
            vibe = np.random.uniform(0.0, 0.1, 128)
            audio = np.random.uniform(0.0, 0.2, (224, 224, 3))
            vision = np.random.uniform(0.0, 0.2, (224, 224, 3))
        else:
            # Faulty Data - lighter under DHT11
            temp = np.random.normal(60.0, 3.0)
            
            # Faulty vibe has massive spikes at certain frequencies
            vibe = np.random.uniform(0.0, 0.3, 128)
            vibe[30:40] += np.random.uniform(0.8, 1.0, 10) 
            
            # Faulty audio and vision have higher intensity/noise
            audio = np.random.uniform(0.5, 1.0, (224, 224, 3))
            vision = np.random.uniform(0.5, 1.0, (224, 224, 3))
            
        temp_data.append([temp])
        vibe_data.append(vibe)
        audio_data.append(audio)
        vision_data.append(vision)
        labels.append([is_faulty])
        
    return (np.array(audio_data, dtype=np.float32), 
            np.array(vision_data, dtype=np.float32), 
            np.array(vibe_data, dtype=np.float32), 
            np.array(temp_data, dtype=np.float32)), np.array(labels, dtype=np.float32)

def main():
    print("=== AURA Multimodal AI Training Module ===")
    model = build_model()
    
    # Generate Data
    X, Y = generate_synthetic_data(num_samples=100) # Small dataset for fast hackathon training
    
    print("Compiling model...")
    # Train
    print("Starting Training (5 epochs)...")
    model.fit([X[0], X[1], X[2], X[3]], Y, epochs=5, batch_size=16, validation_split=0.2)
    
    print("Saving model weights...")
    model.save_weights('aura_weights.weights.h5')
    print("Successfully saved 'aura_weights.weights.h5'!")

if __name__ == "__main__":
    main()
