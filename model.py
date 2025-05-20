import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_dqn_model(input_shape, action_size, learning_rate=0.0001):
    # Disable TensorFlow warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    model = Sequential([
        # First convolutional layer (simplified)
        Conv2D(
            filters=16,  # Reduced from 32
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape
        ),
        
        # Second convolutional layer (simplified)
        Conv2D(
            filters=32,  # Reduced from 64
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu'
        ),
        
        # Flatten the convolutional output
        Flatten(),
        
        # Fully connected layer (simplified)
        Dense(
            units=256,  # Reduced from 512
            activation='relu'
        ),
        
        # Output layer
        Dense(
            units=action_size,
            activation='linear'
        )
    ])
    
    # Compile the model with Mean Squared Error loss
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    return model