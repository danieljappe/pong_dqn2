import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Force CPU only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def build_dqn_model(input_shape, action_size, learning_rate=0.00025):
    """
    Build a CPU-friendly CNN model for DQN
    """
    # Disable TensorFlow warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Use CPU-optimized model with slightly more capacity
    model = Sequential([
        # First convolutional layer
        Conv2D(
            filters=32,  # Increased from 16
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape
        ),
        
        # Second convolutional layer
        Conv2D(
            filters=64,  # Increased from 32
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu'
        ),
        
        # Flatten the convolutional output
        Flatten(),
        
        # Fully connected layer
        Dense(
            units=256,
            activation='relu'
        ),
        
        # Output layer
        Dense(
            units=action_size,
            activation='linear'
        )
    ])
    
    # Compile the model
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    # Print model summary
    model.summary()
    
    return model