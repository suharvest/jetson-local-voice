"""Python ctypes wrapper for the Zipformer encoder engine (libzipformer_encoder.so).

Supports both non-streaming and streaming modes.

Non-streaming usage:
    enc = ZipformerEncoderEngine("/path/to/weights", max_T=64)
    output = enc.run(features)  # features: [T, 256] float32
    # output: [T_out, 512] float32
    enc.close()

Streaming usage:
    enc = ZipformerEncoderEngine("/path/to/weights", max_T=64)
    state = enc.create_state()
    for chunk in chunks_16_frames:
        out = enc.run_chunk(state, chunk)  # chunk: [16, 256], out: [8, 512]
        all_outputs.append(out)
    enc.destroy_state(state)
    enc.close()
"""

import ctypes
import os
import numpy as np


class ZipformerEncoderEngine:
    """Wrapper around the C/NPU Zipformer encoder engine.

    The engine expects post-embed features [T, 256] as input (hidden_dim=256).
    The encoder_embed (Conv2d stack + linear) is NOT included in this engine;
    run it separately (e.g. via ONNX or CPU) to convert [T, 80] fbank
    features to [T', 256] hidden states.

    Output is [T, 512] encoder states ready for the joiner/decoder.
    """

    HIDDEN_DIM = 256
    OUTPUT_DIM = 512

    def __init__(self, weight_dir: str, max_T: int = 64,
                 lib_path: str = None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__),
                                    "libzipformer_encoder.so")

        self.lib = ctypes.CDLL(lib_path)

        # zipformer_encoder_init
        self.lib.zipformer_encoder_init.argtypes = [
            ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.zipformer_encoder_init.restype = ctypes.c_void_p

        # zipformer_encoder_run (non-streaming)
        self.lib.zipformer_encoder_run.argtypes = [
            ctypes.c_void_p,                       # enc
            ctypes.POINTER(ctypes.c_float),         # features
            ctypes.c_int,                           # T
            ctypes.c_int,                           # feat_dim
            ctypes.POINTER(ctypes.c_float),         # output
            ctypes.POINTER(ctypes.c_int),           # out_T
        ]
        self.lib.zipformer_encoder_run.restype = ctypes.c_int

        # zipformer_encoder_destroy
        self.lib.zipformer_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.zipformer_encoder_destroy.restype = None

        # --- Streaming API ---
        # zipformer_state_create
        self.lib.zipformer_state_create.argtypes = [ctypes.c_void_p]
        self.lib.zipformer_state_create.restype = ctypes.c_void_p

        # zipformer_state_reset
        self.lib.zipformer_state_reset.argtypes = [ctypes.c_void_p]
        self.lib.zipformer_state_reset.restype = None

        # zipformer_state_destroy
        self.lib.zipformer_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.zipformer_state_destroy.restype = None

        # zipformer_encoder_run_chunk
        self.lib.zipformer_encoder_run_chunk.argtypes = [
            ctypes.c_void_p,                       # enc
            ctypes.c_void_p,                       # state
            ctypes.POINTER(ctypes.c_float),         # chunk_features
            ctypes.c_int,                           # T_chunk
            ctypes.c_int,                           # feat_dim
            ctypes.POINTER(ctypes.c_float),         # output
            ctypes.POINTER(ctypes.c_int),           # out_T
        ]
        self.lib.zipformer_encoder_run_chunk.restype = ctypes.c_int

        # Initialize
        self._max_T = max_T
        self._engine = self.lib.zipformer_encoder_init(
            weight_dir.encode("utf-8"), max_T
        )
        if not self._engine:
            raise RuntimeError(
                f"zipformer_encoder_init failed. Check weight_dir={weight_dir}"
            )

    def run(self, features: np.ndarray) -> np.ndarray:
        """Run the encoder on post-embed features (non-streaming).

        Args:
            features: [T, 256] float32 hidden states (post encoder_embed)

        Returns:
            output: [T_out, 512] float32 encoder output
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        assert features.ndim == 2, f"Expected 2D input, got {features.ndim}D"
        T, feat_dim = features.shape
        assert feat_dim == self.HIDDEN_DIM, \
            f"feat_dim={feat_dim}, expected {self.HIDDEN_DIM}"
        assert T <= self._max_T, \
            f"T={T} exceeds max_T={self._max_T}"

        features = np.ascontiguousarray(features, dtype=np.float32)
        max_out_T = (T + 1) // 2
        output = np.zeros((max_out_T, self.OUTPUT_DIM), dtype=np.float32)
        out_T = ctypes.c_int(0)

        ret = self.lib.zipformer_encoder_run(
            self._engine,
            features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            T,
            feat_dim,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_T),
        )
        if ret != 0:
            raise RuntimeError(f"zipformer_encoder_run failed with code {ret}")

        return output[:out_T.value]

    # --- Streaming API ---

    def create_state(self):
        """Create a new streaming state (zero-initialized).

        Returns:
            state: opaque ctypes handle
        """
        state = self.lib.zipformer_state_create(self._engine)
        if not state:
            raise RuntimeError("zipformer_state_create failed")
        return state

    def reset_state(self, state):
        """Reset streaming state to initial (zero) values."""
        self.lib.zipformer_state_reset(state)

    def destroy_state(self, state):
        """Destroy streaming state and free resources."""
        if state:
            self.lib.zipformer_state_destroy(state)

    def run_chunk(self, state, chunk_features: np.ndarray) -> np.ndarray:
        """Run one chunk through the streaming encoder.

        Args:
            state: streaming state handle from create_state()
            chunk_features: [T_chunk, 256] float32 (post-embed, typically T_chunk=16)

        Returns:
            output: [out_T, 512] float32 (typically out_T=8)
        """
        if chunk_features.ndim == 1:
            chunk_features = chunk_features.reshape(1, -1)
        assert chunk_features.ndim == 2
        T_chunk, feat_dim = chunk_features.shape
        assert feat_dim == self.HIDDEN_DIM

        chunk_features = np.ascontiguousarray(chunk_features, dtype=np.float32)
        max_out_T = (T_chunk + 1) // 2
        output = np.zeros((max_out_T, self.OUTPUT_DIM), dtype=np.float32)
        out_T = ctypes.c_int(0)

        ret = self.lib.zipformer_encoder_run_chunk(
            self._engine,
            state,
            chunk_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            T_chunk,
            feat_dim,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_T),
        )
        if ret != 0:
            raise RuntimeError(f"zipformer_encoder_run_chunk failed: {ret}")

        return output[:out_T.value]

    def close(self):
        if self._engine:
            self.lib.zipformer_encoder_destroy(self._engine)
            self._engine = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import sys
    import time

    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zipformer_weights"
    mode = sys.argv[2] if len(sys.argv) > 2 else "nonstream"
    max_T = 64

    print(f"Weight dir: {weight_dir}")
    print(f"Mode: {mode}")

    t0 = time.time()
    enc = ZipformerEncoderEngine(weight_dir, max_T=max_T)
    print(f"Init: {(time.time() - t0)*1000:.0f}ms")

    rng = np.random.RandomState(42)

    if mode == "stream":
        # Streaming mode: process multiple 16-frame chunks
        n_chunks = 4
        chunk_T = 16
        print(f"\nStreaming mode: {n_chunks} chunks of T={chunk_T}")

        state = enc.create_state()
        all_outputs = []

        for i in range(n_chunks):
            chunk = (rng.randn(chunk_T, 256) * 0.1).astype(np.float32)
            t0 = time.time()
            out = enc.run_chunk(state, chunk)
            elapsed = (time.time() - t0) * 1000
            print(f"  Chunk {i}: input [{chunk_T}, 256] -> output {out.shape} "
                  f"({elapsed:.1f}ms)")
            all_outputs.append(out)

        encoder_out = np.concatenate(all_outputs, axis=0)
        print(f"\nTotal output: {encoder_out.shape}")
        print(f"Range: [{encoder_out.min():.6f}, {encoder_out.max():.6f}]")
        print(f"Mean: {encoder_out.mean():.6f}, Std: {encoder_out.std():.6f}")

        enc.destroy_state(state)
    else:
        # Non-streaming mode
        test_T = 16
        features = (rng.randn(test_T, 256) * 0.1).astype(np.float32)

        t0 = time.time()
        output = enc.run(features)
        elapsed = (time.time() - t0) * 1000
        print(f"\nOutput shape: {output.shape}")
        print(f"Range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Mean: {output.mean():.6f}, Std: {output.std():.6f}")
        print(f"Inference: {elapsed:.1f}ms")

    enc.close()
