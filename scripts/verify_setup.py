"""
Environment Verification Script
================================
Validates that all required packages, hardware, and configurations
are properly set up for the data augmentation pipeline.

Run: python scripts/verify_setup.py
"""

import sys
import os
import platform
from pathlib import Path

# Force UTF-8 output on Windows to render checkmarks and symbols
if sys.platform == "win32":
    os.system("")  # Enable ANSI/VT100 escape sequences
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def check_python_version() -> bool:
    """Verify Python 3.10+ is installed."""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    status = "✓" if ok else "✗"
    print(f"  {status} Python {version.major}.{version.minor}.{version.micro}", end="")
    if ok:
        print(" (>= 3.10 required)")
    else:
        print(" (FAIL: need >= 3.10)")
    return ok


def check_pytorch() -> bool:
    """Verify PyTorch 2.x with CUDA 11.8."""
    try:
        import torch
        version_ok = int(torch.__version__.split(".")[0]) >= 2
        cuda_ok = torch.cuda.is_available()
        cuda_version = torch.version.cuda if torch.version.cuda else "N/A"

        status_v = "✓" if version_ok else "✗"
        print(f"  {status_v} PyTorch {torch.__version__}", end="")
        if not version_ok:
            print(" (FAIL: need >= 2.0)")
        else:
            print()

        status_c = "✓" if cuda_ok else "✗"
        print(f"  {status_c} CUDA available: {cuda_ok} (version: {cuda_version})")

        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            print(f"  ✓ GPU: {gpu_name}")
            print(f"  ✓ VRAM: {vram_total:.1f} GB")
        else:
            print("  ⚠ No GPU detected — training will be slow on CPU")

        return version_ok  # CUDA is nice-to-have for verification
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_package(name: str, import_name: str | None = None) -> bool:
    """Verify a package is importable and print its version."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {name} {version}")
        return True
    except ImportError:
        print(f"  ✗ {name} not installed")
        return False


def check_gpu_tensor_test() -> bool:
    """Create a test tensor on GPU to verify no OOM on a typical batch."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠ Skipping GPU tensor test (no CUDA)")
            return True

        # Simulate a vision batch: 32 × 3 × 224 × 224 float32
        print("  → Creating test tensor: 32×3×224×224 on GPU...")
        tensor = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float32)
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"  ✓ Tensor created successfully ({mem_allocated:.1f} MB allocated)")

        # Cleanup
        del tensor
        torch.cuda.empty_cache()
        print("  ✓ GPU memory released")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ✗ OOM Error: {e}")
        else:
            print(f"  ✗ GPU tensor test failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ GPU tensor test failed: {e}")
        return False


def check_vram_details() -> None:
    """Print detailed VRAM information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return

        props = torch.cuda.get_device_properties(0)
        total = props.total_mem / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

        print(f"\n  GPU Memory Details:")
        print(f"    Total VRAM:     {total:.2f} GB")
        print(f"    Allocated:      {allocated:.4f} GB")
        print(f"    Reserved:       {reserved:.4f} GB")
        print(f"    Free (approx):  {total - reserved:.2f} GB")
        print(f"    Compute Cap:    {props.major}.{props.minor}")
        print(f"    SM Count:       {props.multi_processor_count}")
    except Exception:
        pass


def check_project_structure() -> bool:
    """Verify critical project directories exist."""
    # Resolve project root relative to this script's location
    project_root = Path(__file__).resolve().parent.parent
    required_dirs = [
        "config",
        "src",
        "src/augmentation",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
    ]

    all_ok = True
    for d in required_dirs:
        path = project_root / d
        exists = path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {d}/")
        if not exists:
            all_ok = False

    # Create data directories if they don't exist (they're gitignored)
    data_dirs = [
        "data/raw/cifar10",
        "data/raw/sst2",
        "data/augmented/vision",
        "data/augmented/nlp",
        "experiments/results/vision",
        "experiments/results/nlp",
        "reports/figures/vision",
        "reports/figures/nlp",
    ]
    print("\n  Creating data/output directories...")
    for d in data_dirs:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")

    return all_ok


def main() -> None:
    """Run all environment checks."""
    print("=" * 60)
    print("  DATA AUGMENTATION PIPELINE — Environment Verification")
    print("=" * 60)

    print(f"\n  System: {platform.system()} {platform.release()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")

    results: dict[str, bool] = {}

    # Python version
    print("\n[1/6] Python Version")
    results["python"] = check_python_version()

    # PyTorch + CUDA
    print("\n[2/6] PyTorch & CUDA")
    results["pytorch"] = check_pytorch()

    # Core packages
    print("\n[3/6] Core Packages")
    packages = [
        ("albumentations", "albumentations"),
        ("nlpaug", "nlpaug"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml"),
        ("Pillow", "PIL"),
        ("scipy", "scipy"),
    ]
    for name, import_name in packages:
        results[name] = check_package(name, import_name)

    # GPU tensor test
    print("\n[4/6] GPU Tensor Test")
    results["gpu_test"] = check_gpu_tensor_test()

    # VRAM details
    print("\n[5/6] VRAM Details")
    check_vram_details()

    # Project structure
    print("\n[6/6] Project Structure")
    results["structure"] = check_project_structure()

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    all_ok = all(results.values())

    print("\n" + "=" * 60)
    if all_ok:
        print(f"  ✓ ALL CHECKS PASSED ({passed}/{total})")
        print("  Ready to run the pipeline!")
    else:
        print(f"  ⚠ {passed}/{total} checks passed")
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed: {', '.join(failed)}")
        print("  Fix the above issues before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
