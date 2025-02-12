import torch
import gc
from typing import Tuple
import psutil
import os

def get_memory_info() -> Tuple[float, float, float, float]:
    """Get system and GPU memory information"""
    # System memory
    system_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # GB
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        gpu_max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
    else:
        gpu_memory_allocated = gpu_memory_reserved = gpu_max_memory = 0.0
        
    return system_memory, gpu_memory_allocated, gpu_memory_reserved, gpu_max_memory

def test_cuda_setup():
    """Test PyTorch CUDA setup and perform basic operations"""
    print("\n=== PyTorch CUDA Setup Test ===\n")
    
    # Version and CUDA information
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Memory information before operations
    sys_mem, gpu_alloc, gpu_reserved, gpu_max = get_memory_info()
    print(f"\nInitial Memory State:")
    print(f"System Memory Usage: {sys_mem:.2f} GB")
    print(f"GPU Memory Allocated: {gpu_alloc:.2f} GB")
    print(f"GPU Memory Reserved: {gpu_reserved:.2f} GB")
    print(f"GPU Total Memory: {gpu_max:.2f} GB")
    
    if torch.cuda.is_available():
        try:
            # Test basic tensor operations
            print("\nTesting CUDA Tensor Operations...")
            
            # Create large tensors to test memory handling
            size = (5000, 5000)
            print(f"Creating tensors of size {size}...")
            
            # Test FP32
            x = torch.randn(*size, device='cuda')
            y = torch.randn(*size, device='cuda')
            z = torch.matmul(x, y)
            del z
            print("✓ FP32 matrix multiplication successful")
            
            # Test FP16 (mixed precision)
            x = x.half()
            y = y.half()
            z = torch.matmul(x, y)
            del z
            print("✓ FP16 matrix multiplication successful")
            
            # Clear memory
            del x, y
            torch.cuda.empty_cache()
            gc.collect()
            
            # Memory information after operations
            sys_mem, gpu_alloc, gpu_reserved, gpu_max = get_memory_info()
            print(f"\nFinal Memory State:")
            print(f"System Memory Usage: {sys_mem:.2f} GB")
            print(f"GPU Memory Allocated: {gpu_alloc:.2f} GB")
            print(f"GPU Memory Reserved: {gpu_reserved:.2f} GB")
            print(f"GPU Total Memory: {gpu_max:.2f} GB")
            
            print("\n✓ All CUDA tests passed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during CUDA testing: {str(e)}")
    else:
        print("\n❌ CUDA is not available. Please check your PyTorch installation.")

if __name__ == "__main__":
    test_cuda_setup()
