#!/usr/bin/env python3
"""Patch sherpa-onnx CUDA wheel to use system onnxruntime 1.20.0.

The sherpa-onnx +cuda aarch64 wheel bundles onnxruntime 1.11.0 (CUDA 10/11).
On Jetson (CUDA 12.6), we need the system's onnxruntime 1.20.0 with
CUDAExecutionProvider. This script:

1. Uses patchelf to change NEEDED from libonnxruntime.so.1.11.0
   to libonnxruntime.so.1 (system SONAME)
2. Patches the ELF version string from VERS_1.11.0 to VERS_1.20.0
3. Patches the ELF version hash to match the new version string
4. Removes the bundled onnxruntime libraries
"""
import glob
import os
import struct
import subprocess


def elf_hash(name: str) -> int:
    """Compute ELF hash for a symbol/version name."""
    h = 0
    for ch in name.encode():
        h = ((h << 4) + ch) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


SHERPA_LIB = "/usr/local/lib/python3.10/dist-packages/sherpa_onnx/lib"
OLD_VER = b"VERS_1.11.0"
NEW_VER = b"VERS_1.20.0"
OLD_HASH = struct.pack("<I", elf_hash("VERS_1.11.0"))
NEW_HASH = struct.pack("<I", elf_hash("VERS_1.20.0"))

# Step 1: patchelf to change NEEDED library reference
for f in [
    "_sherpa_onnx.cpython-310-aarch64-linux-gnu.so",
    "libsherpa-onnx-c-api.so",
    "libsherpa-onnx-cxx-api.so",
]:
    path = os.path.join(SHERPA_LIB, f)
    if os.path.isfile(path):
        subprocess.run(
            [
                "patchelf",
                "--replace-needed",
                "libonnxruntime.so.1.11.0",
                "libonnxruntime.so.1",
                path,
            ],
            check=True,
        )
        print(f"patchelf: {f}")

# Step 2: Patch version string + hash
for so_file in glob.glob(os.path.join(SHERPA_LIB, "*.so")):
    basename = os.path.basename(so_file)
    if "libonnx" in basename:
        continue
    with open(so_file, "rb") as fh:
        data = fh.read()

    ver_count = data.count(OLD_VER)
    hash_count = data.count(OLD_HASH)

    if ver_count > 0 or hash_count > 0:
        if ver_count > 0:
            data = data.replace(OLD_VER, NEW_VER)
        if hash_count > 0:
            data = data.replace(OLD_HASH, NEW_HASH)
        with open(so_file, "wb") as fh:
            fh.write(data)
        print(f"Patched {basename}: ver={ver_count}, hash={hash_count}")

# Step 3: Remove bundled onnxruntime libs (use system's CUDA-enabled ones)
for lib in [
    "libonnxruntime.so",
    "libonnxruntime.so.1.11.0",
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_tensorrt.so",
]:
    path = os.path.join(SHERPA_LIB, lib)
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed bundled: {lib}")

print("Done! sherpa-onnx now uses system onnxruntime 1.20.0 with CUDA.")
