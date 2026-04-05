"""Python ctypes wrapper for the C code predictor engine (libcp_engine.so)."""

import ctypes
import os
import numpy as np


class CodePredictorEngine:
    """Wrapper around the C/NPU code predictor engine.

    Usage:
        engine = CodePredictorEngine("/path/to/cp_weights", num_npu_cores=2)
        codes, codec_sum = engine.run(last_hidden, primary_embed)
    """

    def __init__(self, weight_dir: str, num_npu_cores: int = 2,
                 lib_path: str = None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libcp_engine.so")

        self.lib = ctypes.CDLL(lib_path)

        # cp_engine_init
        self.lib.cp_engine_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.cp_engine_init.restype = ctypes.c_void_p

        # cp_engine_run
        self.lib.cp_engine_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.cp_engine_run.restype = ctypes.c_int

        # cp_engine_destroy
        self.lib.cp_engine_destroy.argtypes = [ctypes.c_void_p]
        self.lib.cp_engine_destroy.restype = None

        # Initialize engine
        self._engine = self.lib.cp_engine_init(
            weight_dir.encode("utf-8"), num_npu_cores
        )
        if not self._engine:
            raise RuntimeError(
                f"cp_engine_init failed. Check weight_dir={weight_dir}"
            )

    def run(self, last_hidden: np.ndarray, primary_embed: np.ndarray):
        """Run 15-step code predictor.

        Args:
            last_hidden: [1024] float32 - hidden state from main model
            primary_embed: [1024] float32 - primary code embedding

        Returns:
            codes: [15] int32 - predicted codec codes
            codec_sum: [1024] float32 - sum of codec embeddings
        """
        assert last_hidden.shape == (1024,) and last_hidden.dtype == np.float32
        assert primary_embed.shape == (1024,) and primary_embed.dtype == np.float32

        # Ensure contiguous
        last_hidden = np.ascontiguousarray(last_hidden)
        primary_embed = np.ascontiguousarray(primary_embed)

        codes = np.zeros(15, dtype=np.int32)
        codec_sum = np.zeros(1024, dtype=np.float32)

        ret = self.lib.cp_engine_run(
            self._engine,
            last_hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            primary_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            codec_sum.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if ret != 0:
            raise RuntimeError(f"cp_engine_run failed with code {ret}")

        return codes, codec_sum

    def close(self):
        if self._engine:
            self.lib.cp_engine_destroy(self._engine)
            self._engine = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
