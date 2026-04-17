import tensorflow as tf
import cv2
import numpy as np

def compute_gradcam(model, inputs_list, layer_name='vision_last_conv'):
    """
    Computes Grad-CAM heatmap using TensorFlow GradientTape on the specified layer.
    """
    # Create a model that maps the input image to the activations of the last conv layer 
    # as well as the output predictions.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs_list)
        # Assuming predictions is shape (batch_size, 1), we want the 0th index
        loss = predictions[:, 0]

    # Calculate gradients of the loss w.r.t the feature map
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    # Weight the feature maps by the pooled gradients
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU activation
    heatmap = tf.maximum(heatmap, 0) 
    
    # Normalize between 0 and 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def overlay_gradcam(img, heatmap):
    """
    Overlays the Grad-CAM heatmap on the original video frame.
    """
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to 8-bit image and apply HOT colormap (red/orange, no blue)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    
    # Superimpose heatmap onto the original image (lighter blend)
    superimposed_img = cv2.addWeighted(img, 0.75, heatmap, 0.25, 0)
    return superimposed_img
