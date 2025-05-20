import os
import sys
import platform
import subprocess
import tensorflow as tf

def check_gpu_details():
    """Check detailed GPU information"""
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    # Check GPU devices recognized by TensorFlow
    print("\n=== TensorFlow GPU Detection ===")
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPU devices: {physical_devices}")
    
    # Check for CUDA availability
    print("\n=== CUDA Environment Variables ===")
    for var in ['CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    try:
        # Try to get GPU info via nvidia-smi
        print("\n=== NVIDIA System Management Interface ===")
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi error:", result.stderr)
    except:
        print("nvidia-smi command not found or failed to execute.")
    
    # Check CUDA installation
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("\n=== CUDA Compiler Version ===")
        print(result.stdout)
    except:
        print("\nCUDA compiler (nvcc) not found in PATH.")
    
    # Check TensorFlow CUDA build information
    print("\n=== TensorFlow CUDA Build Info ===")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU device name: {tf.test.gpu_device_name() if hasattr(tf.test, 'gpu_device_name') else 'Function not available'}")
    
    # Try to execute a simple operation on GPU
    print("\n=== GPU Execution Test ===")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print("GPU execution successful")
    except Exception as e:
        print(f"GPU execution failed: {e}")
    
    # Check TensorFlow's ability to access the GPU
    print("\n=== TensorFlow GPU Configuration ===")
    try:
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
    except Exception as e:
        print(f"Error getting device library: {e}")
    
    # Check for DirectML possibility (Windows)
    if platform.system() == 'Windows':
        print("\n=== DirectML Alternative ===")
        try:
            import win32com.client
            wmi = win32com.client.GetObject("winmgmts:")
            for gpu in wmi.InstancesOf("Win32_VideoController"):
                print(f"Video card: {gpu.Name}")
            print("You could try tensorflow-directml as an alternative if CUDA setup is problematic.")
        except:
            print("Could not check video controllers through WMI.")
            try:
                result = subprocess.run(['dxdiag', '/t', 'dxdiag_output.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("DxDiag run - check dxdiag_output.txt for GPU information")
            except:
                print("Could not run dxdiag.")

if __name__ == "__main__":
    check_gpu_details()
import os
import sys
import platform
import subprocess
import tensorflow as tf

def check_gpu_details():
    """Check detailed GPU information"""
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    # Check GPU devices recognized by TensorFlow
    print("\n=== TensorFlow GPU Detection ===")
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPU devices: {physical_devices}")
    
    # Check for CUDA availability
    print("\n=== CUDA Environment Variables ===")
    for var in ['CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        print("\n=== NVIDIA GPU Information ===")
        device_count = nvidia_smi.nvmlDeviceGetCount()
        print(f"Device count: {device_count}")
        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            name = nvidia_smi.nvmlDeviceGetName(handle)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            driver = nvidia_smi.nvmlSystemGetDriverVersion()
            print(f"GPU {i}: {name.decode('utf-8')}")
            print(f"  Driver Version: {driver.decode('utf-8')}")
            print(f"  Memory Total: {memory.total / 1024**2:.2f} MB")
            print(f"  Memory Used: {memory.used / 1024**2:.2f} MB")
            print(f"  Memory Free: {memory.free / 1024**2:.2f} MB")
        nvidia_smi.nvmlShutdown()
    except:
        print("\nCould not initialize NVIDIA SMI. Trying nvidia-smi command...")
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(result.stdout)
        except:
            print("nvidia-smi command not found.")
    
    # Check CUDA installation
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("\n=== CUDA Compiler Version ===")
        print(result.stdout)
    except:
        print("\nCUDA compiler (nvcc) not found in PATH.")
    
    # Check TensorFlow CUDA build information
    print("\n=== TensorFlow CUDA Build Info ===")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else 'Function deprecated'}")
    
    # Try to execute a simple operation on GPU
    print("\n=== GPU Execution Test ===")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print("GPU execution successful")
    except Exception as e:
        print(f"GPU execution failed: {e}")
        
    # Additional check for cudnn
    if hasattr(tf.test, 'is_built_with_cudnn'):
        print(f"\nBuilt with cuDNN: {tf.test.is_built_with_cudnn()}")

if __name__ == "__main__":
    check_gpu_details()