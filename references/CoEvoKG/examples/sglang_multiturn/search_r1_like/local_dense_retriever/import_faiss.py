def import_faiss():
    import subprocess
    import sys

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "faiss-gpu-cu12==1.8.0.2",
    ])
    print("faiss-gpu-cu12 installed successfully.")


if __name__ == "__main__":
    try:
        import faiss
        # Verify GPU support is available
        _ = faiss.GpuMultipleClonerOptions
    except (ImportError, AttributeError):
        import_faiss()
