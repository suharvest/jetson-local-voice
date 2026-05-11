'''
Scaffold for reproducing TensorRT 10.3 Myelin 'already loaded binary graph' bug.

Symptom in production pipeline:
    [TRT] IExecutionContext::enqueueV3: Error Code 1: Myelin
      ([executor.cpp:myelinGraphLoad:836] Called with an already loaded binary graph.)

LIMITATION: Single-engine two-call sequence does NOT trigger the bug on our
affected platform (JP6.2 + TRT 10.3.0). The real trigger is a cross-engine
pattern (engine A enqueueV3, then engine B enqueueV3 with correlated shapes,
repeated). See accompanying issue.md.

This script is a starting scaffold for NVIDIA engineers to extend with a
second engine; on our platform it currently exits 0 cleanly even on the
affected TRT version.

Usage: python3 repro.py <engine_path>
Exit 0: both enqueueV3 calls succeeded on single engine
Exit 2: setup error (engine failed to deserialize, no dynamic profile, etc.)
Exit 3: second enqueueV3 failed on same shape (bug triggered in single engine)
'''
import tensorrt as trt
from cuda import cuda
import numpy as np
import sys
import os
class StdoutLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self); self.messages = []
    def log(self, severity, msg):
        line = "[TRT] %s: %s" % (str(severity).split(".")[-1], msg)
        self.messages.append(line)
        if "Myelin" in msg or "already loaded binary graph" in msg or "ERROR" in line:
            print(line)
def chk(result, where):
    code, rest = (result[0], result[1:]) if isinstance(result, tuple) else (result, ())
    if int(code) != 0:
        raise RuntimeError("CUDA error %s at %s" % (code, where))
    return None if len(rest) == 0 else rest[0] if len(rest) == 1 else rest
def clean(shape):
    return tuple(1 if int(d) <= 0 else int(d) for d in shape)
def bytes_for(shape, dtype):
    sizes = {trt.float32: 4, trt.float16: 2, trt.bfloat16: 2,
             trt.int32: 4, trt.int64: 8, trt.int8: 1, trt.bool: 1}
    return max(1, int(np.prod(np.array(shape, dtype=np.int64)))) * sizes.get(dtype, 4)
def profile_shapes(engine, name, profile):
    try:
        mn, opt, mx = engine.get_tensor_profile_shape(name, profile)
        return clean(mn), clean(opt), clean(mx)
    except Exception:
        try:
            s = clean(engine.get_tensor_shape(name)); return s, s, s
        except Exception:
            return None
def set_shapes(ctx, inputs, key):
    for t in inputs:
        ok = ctx.set_input_shape(t["name"], t[key])
        if ok is False:
            raise RuntimeError("set_input_shape failed for %s" % t["name"])
    if hasattr(ctx, "infer_shapes"):
        missing = ctx.infer_shapes()
        if missing:
            print("infer_shapes missing tensors: %s" % list(missing))
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 %s <engine_path>" % os.path.basename(sys.argv[0])); return 2
    print("TRT version: %s" % trt.__version__); print("Engine path: %s" % sys.argv[1])
    chk(cuda.cuInit(0), "cuInit"); dev = chk(cuda.cuDeviceGet(0), "cuDeviceGet")
    cu_ctx = chk(cuda.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
    chk(cuda.cuCtxSetCurrent(cu_ctx), "cuCtxSetCurrent")
    stream = chk(cuda.cuStreamCreate(0), "cuStreamCreate")
    logger = StdoutLogger()
    with open(sys.argv[1], "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        print("deserialize failed"); return 2
    print("engine metadata: num_io_tensors=%d num_profiles=%d" % (engine.num_io_tensors, engine.num_optimization_profiles))
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        t = {"name": name, "mode": engine.get_tensor_mode(name), "dtype": engine.get_tensor_dtype(name)}
        (inputs if t["mode"] == trt.TensorIOMode.INPUT else outputs).append(t)
    profile = None
    # Score each profile by count of inputs whose max shape has a non-zero
    # dynamic dim. Decode-like profiles (past_key has max>0 varying) score
    # higher than prefill profiles (where past=0 fixed). Pick highest score.
    best_score = -1
    for p in range(engine.num_optimization_profiles):
        score = 0
        for t in inputs:
            s = profile_shapes(engine, t["name"], p)
            if not s:
                continue
            # Count dims that are dynamic with non-zero max
            for mn, mx in zip(s[0], s[2]):
                if mn != mx and mx > 0:
                    score += 1
            # Penalize profiles where any 4D input has max=0 on any dim
            # (classic prefill-with-empty-KV pattern)
            for mx_dim in s[2]:
                if len(s[2]) >= 4 and mx_dim == 0:
                    score -= 10
        if score > best_score:
            best_score = score; profile = p
    if profile is None:
        print("no dynamic profile found, cannot test"); return 0
    print("selected dynamic profile: %d" % profile)
    for t in inputs:
        t["min"], t["opt"], t["max"] = profile_shapes(engine, t["name"], profile)
    ctx = engine.create_execution_context()
    ok = ctx.set_optimization_profile_async(profile, int(stream))
    if ok is False:
        raise RuntimeError("set_optimization_profile_async failed")
    chk(cuda.cuStreamSynchronize(stream), "profile sync")
    set_shapes(ctx, inputs, "max")
    for t in outputs:
        t["max"] = clean(ctx.get_tensor_shape(t["name"]))
    set_shapes(ctx, inputs, "opt")
    for t in inputs + outputs:
        shape = t["opt"] if t in inputs else t["max"]
        size = bytes_for(shape, t["dtype"])
        ptr = chk(cuda.cuMemAlloc(size), "cuMemAlloc %s" % t["name"])
        chk(cuda.cuMemsetD8(ptr, 0, size), "cuMemsetD8 %s" % t["name"])
        if ctx.set_tensor_address(t["name"], int(ptr)) is False:
            raise RuntimeError("set_tensor_address failed for %s" % t["name"])
        print("tensor %s mode=%s dtype=%s shape=%s bytes=%d" % (t["name"], t["mode"], t["dtype"], shape, size))
    for call in (1, 2):
        start = len(logger.messages); print("enqueueV3 call #%d starting" % call)
        try:
            ok = ctx.execute_async_v3(stream_handle=int(stream))
            chk(cuda.cuStreamSynchronize(stream), "enqueue sync #%d" % call)
        except Exception as exc:
            print("enqueueV3 call #%d exception: %s" % (call, exc)); ok = False
        errs = [m for m in logger.messages[start:] if "Myelin" in m or "ERROR" in m]
        if errs:
            print("enqueueV3 call #%d TRT errors: %s" % (call, " | ".join(errs)))
        print("enqueueV3 call #%d result: %s" % (call, ok))
        if not ok:
            return 3 if call == 2 else 2
    print("both enqueueV3 calls OK; bug NOT reproduced")
    return 0
if __name__ == "__main__":
    sys.exit(main())
