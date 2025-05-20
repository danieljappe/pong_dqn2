import tensorflow as tf

# Set memory growth before anything else
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled")
    except Exception as e:
        print(f"Error setting memory growth: {e}")

# Now try to use the GPU
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", bool(physical_devices))

# Simple test
if len(physical_devices) > 0:
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print("GPU test successful. Result:")
            print(c)
    except Exception as e:
        print(f"GPU test failed: {e}")
else:
    print("No GPU found by TensorFlow")

# Print CUDA/cuDNN info
print("\nCUDA/cuDNN Information:")
print("Built with CUDA:", tf.test.is_built_with_cuda())
if hasattr(tf.test, "is_built_with_cudnn"):
    print("Built with cuDNN:", tf.test.is_built_with_cudnn())