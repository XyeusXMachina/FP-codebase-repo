import torch
import platform
import subprocess

print("=" * 60)
print("SYSTEM")
print("=" * 60)
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

print("\n" + "=" * 60)
print("GPU INFORMATION")
print("=" * 60)

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")

    for i in range(gpu_count):
        gpu = torch.cuda.get_device_properties(i)

        print(f"\nGPU {i}")
        print(f"Name: {gpu.name}")
        print(f"VRAM: {gpu.total_memory / 1024**3:.2f} GB")
        print(f"Compute capability: {gpu.major}.{gpu.minor}")

    print("\n" + "=" * 60)
    print("GPU COMPUTATION TEST")
    print("=" * 60)

    device = torch.device("cuda")

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    torch.cuda.synchronize()

    z = torch.matmul(x, y)

    torch.cuda.synchronize()

    print("Matrix multiplication successful.")
    print(f"Tensor device: {z.device}")
    print(f"Tensor shape: {z.shape}")

    del x, y, z
    torch.cuda.empty_cache()

    print("\nGPU test PASSED.")

else:
    print("CUDA is NOT available.")
    print("PyTorch cannot currently use the GPU.")