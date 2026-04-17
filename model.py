import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
    """
    Builds a 4-input Multimodal Neural Network for Audio, Vision, Vibration, and Temperature.
    """
    # Branch A: Audio (Mel-Spectrogram)
    input_a = Input(shape=(224, 224, 3), name='audio_input')
    x = Conv2D(32, (3, 3), activation='relu')(input_a)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='audio_last_conv')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    flat_a = Flatten()(x)
    
    # Auxiliary output for audio probability
    out_audio = Dense(1, activation='sigmoid', name='audio_prob')(flat_a)

    # Branch B: Vision (Optical Flow)
    input_b = Input(shape=(224, 224, 3), name='vision_input')
    y = Conv2D(32, (3, 3), activation='relu')(input_b)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(128, (3, 3), activation='relu', name='vision_last_conv')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Dropout(0.3)(y)
    flat_b = Flatten()(y)
    
    # Auxiliary output for vision probability
    out_vision = Dense(1, activation='sigmoid', name='vision_prob')(flat_b)

    # Branch C: Vibration (1D FFT Array)
    input_c = Input(shape=(128,), name='vibration_input')
    v = Dense(64, activation='relu')(input_c)
    v = Dropout(0.2)(v)
    v = Dense(32, activation='relu')(v)

    # Branch D: Temperature (Scalar)
    input_d = Input(shape=(1,), name='temp_input')
    t = Dense(16, activation='relu')(input_d)

    # Merge all 4 branches via Concatenate layer
    merged = Concatenate()([flat_a, flat_b, v, t])
    
    # Dense layers for final decision
    z = Dense(256, activation='relu')(merged)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    out_fused = Dense(1, activation='sigmoid', name='fused_prob')(z)

    model = Model(inputs=[input_a, input_b, input_c, input_d], outputs=out_fused)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_model(model, filepath='aura_model.h5'):
    """Saves the trained model to disk."""
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='aura_model.h5'):
    """Loads a trained model from disk."""
    return tf.keras.models.load_model(filepath)

def train_model():
    """
    Training script with data augmentation (random flip, brightness, rotation).
    """
    print("Initializing training script...")
    
    # Data Augmentation Setup
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        rotation_range=20
    )
    
    print("Data generator configured.")
    print("In a real scenario, your data generator would yield batches of:")
    print("X: [audio_batch, vision_batch, vibration_batch, temp_batch]")
    print("Y: labels_batch")
    # e.g. datagen.flow_from_directory(...) or tf.data.Dataset
    # model.fit(...)

if __name__ == "__main__":
    # Test model building
    model = build_model()
    model.summary()
