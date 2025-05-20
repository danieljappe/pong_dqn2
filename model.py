import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_dqn_model(input_shape, action_size, learning_rate=0.00025):
    """
    Build a CNN-based DQN model optimized for CPU training
    """
    # Disable TensorFlow warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Check if running on CPU and optimize accordingly
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("Running on CPU. Using optimized CPU configuration.")
        # Use more CPU-friendly model with fewer parameters
        model = Sequential([
            # First convolutional layer (simplified for CPU)
            Conv2D(
                filters=16,  # Reduced filters for CPU
                kernel_size=(8, 8),
                strides=(4, 4),
                activation='relu',
                input_shape=input_shape
            ),
            
            # Second convolutional layer
            Conv2D(
                filters=32,  # Reduced filters for CPU
                kernel_size=(4, 4),
                strides=(2, 2),
                activation='relu'
            ),
            
            # Flatten the convolutional output
            Flatten(),
            
            # Fully connected layer (reduced size)
            Dense(
                units=256,  # Smaller dense layer for CPU
                activation='relu'
            ),
            
            # Output layer
            Dense(
                units=action_size,
                activation='linear'
            )
        ])
    else:
        # Original model for GPU
        model = Sequential([
            # First convolutional layer
            Conv2D(
                filters=32,
                kernel_size=(8, 8),
                strides=(4, 4),
                activation='relu',
                input_shape=input_shape
            ),
            
            # Second convolutional layer
            Conv2D(
                filters=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                activation='relu'
            ),
            
            # Third convolutional layer
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu'
            ),
            
            # Flatten the convolutional output
            Flatten(),
            
            # Fully connected layer
            Dense(
                units=512,
                activation='relu'
            ),
            
            # Output layer
            Dense(
                units=action_size,
                activation='linear'
            )
        ])
    
    # Use Huber loss instead of MSE for stability
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    # Print model summary
    model.summary()
    
    return modelimport tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_dqn_model(input_shape, action_size, learning_rate=0.00025):
    """
    Build a CNN-based DQN model optimized for Pong with GPU acceleration
    """
    # Disable TensorFlow warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure TensorFlow to use GPU memory efficiently
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception:
                # Already configured or not needed
                pass
        
        # Enable mixed precision for faster training on GPU
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Using mixed precision policy:", policy.name)
        except Exception as e:
            print(f"Could not set mixed precision: {e}")
    
    model = Sequential([
        # First convolutional layer
        Conv2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape  # Should be (84, 84, 4) with frame stacking
        ),
        
        # Second convolutional layer
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu'
        ),
        
        # Third convolutional layer
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='relu'
        ),
        
        # Flatten the convolutional output
        Flatten(),
        
        # Fully connected layer
        Dense(
            units=512,
            activation='relu'
        ),
        
        # Output layer
        Dense(
            units=action_size,
            activation='linear'
        )
    ])
    
    # Use Huber loss instead of MSE for stability
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    return modelimport tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_dqn_model(input_shape, action_size, learning_rate=0.00025):
    """
    Build a CNN-based DQN model optimized for Pong
    """
    # Disable TensorFlow warning messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    model = Sequential([
        # First convolutional layer
        Conv2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape  # Should be (84, 84, 4) with frame stacking
        ),
        
        # Second convolutional layer
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu'
        ),
        
        # Third convolutional layer
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='relu'
        ),
        
        # Flatten the convolutional output
        Flatten(),
        
        # Fully connected layer
        Dense(
            units=512,
            activation='relu'
        ),
        
        # Output layer
        Dense(
            units=action_size,
            activation='linear'
        )
    ])
    
    # Use Huber loss instead of MSE for stability
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    return model