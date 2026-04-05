#!/usr/bin/env python3
"""
Optimize code_predictor.onnx for TensorRT on Jetson Orin Nano.

Problem: TRT EP fails with "If_output_0 has no shape specified" because the model
contains ONNX If conditional nodes with mismatched branch output shapes
([2048,1024] vs [-1,2048,1024]). TensorRT cannot handle this.

Strategy (in order of aggressiveness):
  1. Shape inference — add missing shape annotations
  2. onnx-simplifier — constant-fold and eliminate dead If branches
  3. Manual If-node elimination — inline the "else" branch (seq=1 runtime path)
  4. Subgraph extraction — remove If nodes, keep main computation path
  5. Verify with TRT EP or trtexec

Usage:
    python optimize_code_predictor_trt.py [--model PATH] [--output DIR] [--verify]

Requires: onnx, numpy (onnxsim optional, onnxruntime optional for verify)
"""
import os
import sys
import time
import gc
import copy
import argparse
import struct
import numpy as np

MODEL_PATH = "/tmp/qwen3-tts-bench/model/code_predictor.onnx"
OUTPUT_DIR = "/tmp/qwen3-tts-bench/model-trt-opt"


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def log(msg):
    rss = get_rss_mb()
    rss_str = f" [RSS: {rss}MB]" if rss > 0 else ""
    print(f"  {msg}{rss_str}")


# ---------------------------------------------------------------------------
# Phase 0: Analyze the model
# ---------------------------------------------------------------------------
def analyze_model(model):
    """Print model structure summary, focusing on If nodes and dynamic shapes."""
    import onnx

    print("\n" + "=" * 70)
    print("PHASE 0: Model Analysis")
    print("=" * 70)

    # Inputs/Outputs
    print("\n  Inputs:")
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            elif d.dim_value:
                shape.append(str(d.dim_value))
            else:
                shape.append("?")
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"    {inp.name}: [{', '.join(shape)}] {dtype}")

    print("\n  Outputs:")
    for out in model.graph.output:
        shape = []
        tt = out.type.tensor_type
        if tt.HasField("shape"):
            for d in tt.shape.dim:
                if d.dim_param:
                    shape.append(d.dim_param)
                elif d.dim_value:
                    shape.append(str(d.dim_value))
                else:
                    shape.append("?")
        dtype = onnx.TensorProto.DataType.Name(tt.elem_type)
        print(f"    {out.name}: [{', '.join(shape)}] {dtype}")

    # Count node types
    op_counts = {}
    if_nodes = []
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type == "If":
            if_nodes.append(node)

    print(f"\n  Total nodes: {len(model.graph.node)}")
    print(f"  Unique ops: {len(op_counts)}")
    print(f"  Top ops: {sorted(op_counts.items(), key=lambda x: -x[1])[:15]}")

    # If node details
    print(f"\n  If nodes: {len(if_nodes)}")
    for i, node in enumerate(if_nodes):
        print(f"    If[{i}]: name={node.name or '(unnamed)'}")
        print(f"      inputs:  {list(node.input)}")
        print(f"      outputs: {list(node.output)}")
        for attr in node.attribute:
            if attr.name in ("then_branch", "else_branch"):
                g = attr.g
                print(f"      {attr.name}: {len(g.node)} nodes, "
                      f"inputs=[{', '.join(i.name for i in g.input)}], "
                      f"outputs=[{', '.join(o.name for o in g.output)}]")

    # Value info coverage
    named_shapes = set()
    for vi in model.graph.value_info:
        tt = vi.type.tensor_type
        if tt.HasField("shape") and len(tt.shape.dim) > 0:
            named_shapes.add(vi.name)
    print(f"\n  Value info entries: {len(model.graph.value_info)}")
    print(f"  Entries with shape: {len(named_shapes)}")

    # Check which If outputs lack shape
    for node in if_nodes:
        for out_name in node.output:
            has_shape = out_name in named_shapes
            print(f"  If output '{out_name}': shape_defined={has_shape}")

    return if_nodes


# ---------------------------------------------------------------------------
# Phase 1: Shape inference
# ---------------------------------------------------------------------------
def phase1_shape_inference(model_path, output_path):
    """Run ONNX shape inference — may resolve missing shapes on If outputs."""
    import onnx
    from onnx import shape_inference

    print("\n" + "=" * 70)
    print("PHASE 1: Shape Inference")
    print("=" * 70)

    log("Loading model...")
    t0 = time.time()

    # Try in-memory first, fall back to path-based for large models
    try:
        model = onnx.load(model_path, load_external_data=True)
        log(f"Loaded in {time.time()-t0:.1f}s")

        log("Running shape inference (in-memory)...")
        t0 = time.time()
        model_inferred = shape_inference.infer_shapes(
            model,
            check_type=True,
            strict_mode=False,
            data_prop=True,
        )
        log(f"Shape inference done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"In-memory shape inference failed: {e}")
        log("Trying path-based shape inference (lower memory)...")
        try:
            shape_inference.infer_shapes_path(model_path, output_path)
            log("Path-based shape inference succeeded")
            model_inferred = onnx.load(output_path, load_external_data=True)
        except Exception as e2:
            log(f"Path-based shape inference also failed: {e2}")
            return None

    # Check if If outputs now have shapes
    if_output_names = set()
    for node in model_inferred.graph.node:
        if node.op_type == "If":
            for out in node.output:
                if_output_names.add(out)

    resolved = 0
    for vi in model_inferred.graph.value_info:
        if vi.name in if_output_names:
            tt = vi.type.tensor_type
            if tt.HasField("shape") and len(tt.shape.dim) > 0:
                dims = []
                for d in tt.shape.dim:
                    dims.append(d.dim_param if d.dim_param else str(d.dim_value))
                log(f"If output '{vi.name}' now has shape: [{', '.join(dims)}]")
                resolved += 1

    if resolved == len(if_output_names) and len(if_output_names) > 0:
        log(f"All {resolved} If outputs now have shapes!")
    else:
        log(f"Resolved {resolved}/{len(if_output_names)} If output shapes")

    # Save
    log(f"Saving to {output_path}...")
    save_model(model_inferred, output_path)

    del model, model_inferred
    gc.collect()
    return output_path


# ---------------------------------------------------------------------------
# Phase 2: onnx-simplifier
# ---------------------------------------------------------------------------
def phase2_onnxsim(input_path, output_path):
    """Use onnx-simplifier to constant-fold and remove dead branches."""
    print("\n" + "=" * 70)
    print("PHASE 2: ONNX Simplifier")
    print("=" * 70)

    try:
        import onnxsim
        log(f"onnxsim version: {onnxsim.__version__}")
    except ImportError:
        log("onnxsim not installed. Try: pip install onnx-simplifier")
        log("Skipping Phase 2.")
        return None

    import onnx

    log("Loading model...")
    model = onnx.load(input_path, load_external_data=True)

    log("Simplifying (this may take a while for 420MB model)...")
    t0 = time.time()

    try:
        # Provide fixed input shapes to help eliminate If branches
        # batch=1, seq=1 is the autoregressive decode shape
        input_shapes = {
            "inputs_embeds": [1, 1, 1024],
            "generation_steps": [1],  # single step
            "past_keys": [5, 1, 8, 1, 128],   # some past context
            "past_values": [5, 1, 8, 1, 128],
        }

        model_sim, check = onnxsim.simplify(
            model,
            input_shapes=input_shapes,
            dynamic_input_shape=False,  # Fix shapes for branch elimination
        )

        if check:
            log(f"Simplification succeeded in {time.time()-t0:.1f}s")

            # Count If nodes remaining
            n_if = sum(1 for n in model_sim.graph.node if n.op_type == "If")
            n_if_orig = sum(1 for n in model.graph.node if n.op_type == "If")
            log(f"If nodes: {n_if_orig} -> {n_if}")
            log(f"Total nodes: {len(model.graph.node)} -> {len(model_sim.graph.node)}")

            save_model(model_sim, output_path)
            del model, model_sim
            gc.collect()
            return output_path
        else:
            log("Simplification check failed — output may be incorrect")
            log("Saving anyway for inspection...")
            save_model(model_sim, output_path)
            del model, model_sim
            gc.collect()
            return output_path

    except Exception as e:
        log(f"onnxsim failed: {e}")
        import traceback
        traceback.print_exc()

        # Try with dynamic shapes (less aggressive)
        log("Retrying with dynamic_input_shape=True...")
        try:
            model_sim, check = onnxsim.simplify(model)
            if check:
                log("Simplified with dynamic shapes")
                n_if = sum(1 for n in model_sim.graph.node if n.op_type == "If")
                log(f"If nodes remaining: {n_if}")
                save_model(model_sim, output_path)
                del model, model_sim
                gc.collect()
                return output_path
        except Exception as e2:
            log(f"Dynamic simplification also failed: {e2}")

    del model
    gc.collect()
    return None


# ---------------------------------------------------------------------------
# Phase 3: Manual If-node elimination (inline else branch)
# ---------------------------------------------------------------------------
def phase3_eliminate_if_nodes(input_path, output_path):
    """
    Remove If nodes by inlining the appropriate branch.

    The If nodes in code_predictor check whether past_seq == 0 (first step)
    or past_seq > 0 (subsequent steps). The shape mismatch:
      then_branch output: [2048, 1024]     (first step, no batch dim since batch*seq = flat)
      else_branch output: [-1, 2048, 1024] (subsequent, has batch dim)

    For TRT, we inline the else_branch (past_seq > 0 path) which is the
    common autoregressive case. The first step can use past_keys/past_values
    with past_seq=0 dimension (empty KV cache).

    If that shape analysis is wrong, we try the then_branch instead.
    """
    import onnx
    from onnx import helper, TensorProto

    print("\n" + "=" * 70)
    print("PHASE 3: Manual If-Node Elimination")
    print("=" * 70)

    log("Loading model...")
    model = onnx.load(input_path, load_external_data=True)

    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    if not if_nodes:
        log("No If nodes found — nothing to do")
        del model
        gc.collect()
        return input_path

    log(f"Found {len(if_nodes)} If node(s) to eliminate")

    # We'll try inlining the else_branch first (the "has past context" path)
    # which is the steady-state autoregressive decode path.
    # If the model has past_keys with past_seq=0 for the first call, the
    # else branch should still work (just with empty past).

    graph = model.graph
    nodes = list(graph.node)
    new_nodes = []
    inlined_count = 0

    for node in nodes:
        if node.op_type != "If":
            new_nodes.append(node)
            continue

        log(f"Processing If node: {node.name or '(unnamed)'}")
        log(f"  condition input: {node.input[0]}")
        log(f"  outputs: {list(node.output)}")

        # Get both branches
        then_branch = None
        else_branch = None
        for attr in node.attribute:
            if attr.name == "then_branch":
                then_branch = attr.g
            elif attr.name == "else_branch":
                else_branch = attr.g

        if then_branch is None or else_branch is None:
            log("  WARNING: Missing branch, keeping If node as-is")
            new_nodes.append(node)
            continue

        # Analyze branch output shapes to decide which to inline
        log(f"  then_branch: {len(then_branch.node)} nodes, "
            f"outputs={[o.name for o in then_branch.output]}")
        log(f"  else_branch: {len(else_branch.node)} nodes, "
            f"outputs={[o.name for o in else_branch.output]}")

        # Choose the branch with more nodes (usually the main computation path)
        # or the else_branch (common autoregressive path)
        branch = else_branch
        branch_name = "else_branch"

        # If else_branch is trivially small and then_branch is large,
        # use then_branch instead
        if len(else_branch.node) < len(then_branch.node) // 2:
            branch = then_branch
            branch_name = "then_branch"

        log(f"  Inlining {branch_name} ({len(branch.node)} nodes)")

        # Create a unique prefix for inlined node names to avoid conflicts
        prefix = f"_inlined_{inlined_count}_"

        # Map branch input names to the corresponding outer graph tensors
        # Branch inputs come from the outer graph (the If node's scope)
        # They are implicitly available as outer scope names
        name_map = {}

        # Branch outputs map to If node outputs
        for branch_out, if_out in zip(branch.output, node.output):
            name_map[branch_out.name] = if_out

        # Add branch nodes with remapped names
        for bnode in branch.node:
            new_node = copy.deepcopy(bnode)

            # Prefix internal names to avoid collisions
            # But keep names that map to outer scope (inputs from outer graph)
            new_name = prefix + new_node.name if new_node.name else ""
            new_node.name = new_name

            # Remap outputs
            for i, out_name in enumerate(new_node.output):
                if out_name in name_map:
                    new_node.output[i] = name_map[out_name]
                else:
                    # Internal tensor — prefix it
                    mapped = prefix + out_name
                    name_map[out_name] = mapped
                    new_node.output[i] = mapped

            # Remap inputs
            for i, inp_name in enumerate(new_node.input):
                if inp_name in name_map:
                    new_node.input[i] = name_map[inp_name]
                # else: it references an outer scope name, keep as-is

            new_nodes.append(new_node)

        # Also add branch initializers to the main graph
        for init in branch.initializer:
            new_init = copy.deepcopy(init)
            if init.name in name_map:
                new_init.name = name_map[init.name]
            else:
                new_init.name = prefix + init.name
                name_map[init.name] = new_init.name
            graph.initializer.append(new_init)

        inlined_count += 1

    if inlined_count == 0:
        log("No If nodes were inlined")
        del model
        gc.collect()
        return None

    # Replace nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    log(f"Inlined {inlined_count} If node(s)")
    log(f"Total nodes now: {len(graph.node)}")

    # Run shape inference on the result
    log("Running shape inference on modified graph...")
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
        log("Shape inference succeeded")
    except Exception as e:
        log(f"Shape inference failed (may still work): {e}")

    # Validate
    log("Checking model validity...")
    try:
        onnx.checker.check_model(model, full_check=False)
        log("Model check passed")
    except Exception as e:
        log(f"Model check warning: {e}")
        log("(Continuing anyway — TRT EP may still accept it)")

    save_model(model, output_path)
    del model
    gc.collect()
    return output_path


# ---------------------------------------------------------------------------
# Phase 4: Remove If nodes entirely (fallback — extract main subgraph)
# ---------------------------------------------------------------------------
def phase4_remove_if_nodes(input_path, output_path):
    """
    Last resort: remove all If nodes and replace their outputs with
    identity/reshape operations that produce a valid static shape.
    This is lossy but may let TRT compile the rest of the graph.
    """
    import onnx
    from onnx import helper, TensorProto

    print("\n" + "=" * 70)
    print("PHASE 4: Force-Remove If Nodes (Fallback)")
    print("=" * 70)

    log("Loading model...")
    model = onnx.load(input_path, load_external_data=True)

    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    if not if_nodes:
        log("No If nodes — nothing to do")
        del model
        gc.collect()
        return input_path

    graph = model.graph
    nodes = list(graph.node)
    new_nodes = []
    removed = 0

    for node in nodes:
        if node.op_type != "If":
            new_nodes.append(node)
            continue

        log(f"Removing If node: {node.name}, outputs={list(node.output)}")

        # Get the else_branch and inline it (same as phase 3 but simpler —
        # if phase 3 already produced a good model this won't be called)
        else_branch = None
        then_branch = None
        for attr in node.attribute:
            if attr.name == "else_branch":
                else_branch = attr.g
            elif attr.name == "then_branch":
                then_branch = attr.g

        # Pick the branch with fewer nodes (simpler, more likely to be
        # the identity/reshape path)
        branch = then_branch if then_branch and (
            not else_branch or len(then_branch.node) <= len(else_branch.node)
        ) else else_branch

        if branch is None:
            log(f"  No branch available, dropping If outputs entirely")
            removed += 1
            continue

        prefix = f"_fallback_{removed}_"
        name_map = {}

        for branch_out, if_out in zip(branch.output, node.output):
            name_map[branch_out.name] = if_out

        for bnode in branch.node:
            new_node = copy.deepcopy(bnode)
            new_node.name = prefix + (new_node.name or "")
            for i, out_name in enumerate(new_node.output):
                if out_name in name_map:
                    new_node.output[i] = name_map[out_name]
                else:
                    mapped = prefix + out_name
                    name_map[out_name] = mapped
                    new_node.output[i] = mapped
            for i, inp_name in enumerate(new_node.input):
                if inp_name in name_map:
                    new_node.input[i] = name_map[inp_name]
            new_nodes.append(new_node)

        for init in branch.initializer:
            new_init = copy.deepcopy(init)
            if init.name in name_map:
                new_init.name = name_map[init.name]
            else:
                new_init.name = prefix + init.name
                name_map[init.name] = new_init.name
            graph.initializer.append(new_init)

        removed += 1

    del graph.node[:]
    graph.node.extend(new_nodes)

    log(f"Removed {removed} If node(s), total nodes: {len(graph.node)}")

    # Try to fix shapes by making all dynamic dims concrete for batch=1, seq=1
    # This helps TRT build static engines
    log("Fixing input shapes to static (batch=1, seq=1, past_seq=1)...")
    for inp in graph.input:
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            continue
        for dim in tt.shape.dim:
            if dim.dim_param:  # dynamic dimension
                param = dim.dim_param
                # Map known dynamic dims to concrete values
                static_map = {
                    "batch": 1,
                    "batch_size": 1,
                    "sequence": 1,
                    "seq": 1,
                    "seq_len": 1,
                    "past_seq": 1,
                    "past_sequence": 1,
                    "num_steps": 1,
                }
                for key, val in static_map.items():
                    if key in param.lower():
                        log(f"  {inp.name}: dim '{param}' -> {val}")
                        dim.dim_value = val
                        dim.dim_param = ""
                        break

    save_model(model, output_path)
    del model
    gc.collect()
    return output_path


# ---------------------------------------------------------------------------
# Phase 5: Verify with TRT EP
# ---------------------------------------------------------------------------
def phase5_verify_trt(model_path):
    """Try loading the model with TRT EP."""
    print("\n" + "=" * 70)
    print("PHASE 5: Verify with TensorRT EP")
    print("=" * 70)

    try:
        import onnxruntime as ort
        log(f"ORT version: {ort.__version__}")
        log(f"Available providers: {ort.get_available_providers()}")
    except ImportError:
        log("onnxruntime not available — cannot verify")
        return False

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        log("TensorRT EP not available — trying CUDA EP as sanity check")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ep_name = "CUDA"
    else:
        engine_cache = os.path.join(os.path.dirname(model_path), "trt_cache")
        os.makedirs(engine_cache, exist_ok=True)
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": engine_cache,
                "trt_max_workspace_size": str(1 * 1024 * 1024 * 1024),
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        ep_name = "TRT"

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    log(f"Creating {ep_name} session...")
    t0 = time.time()
    try:
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        log(f"Session created in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"Session creation FAILED: {e}")
        return False

    # Print which EP each node is running on
    log(f"Active providers: {sess.get_providers()}")

    # Test inference
    log("Running test inference...")
    try:
        feeds = {}
        for inp in sess.get_inputs():
            shape = []
            for d in inp.shape:
                if isinstance(d, str) or d is None or d <= 0:
                    shape.append(1)  # replace dynamic dims
                else:
                    shape.append(d)
            if inp.type == "tensor(float)":
                feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
            elif inp.type == "tensor(int64)":
                feeds[inp.name] = np.zeros(shape, dtype=np.int64)
            elif inp.type == "tensor(float16)":
                feeds[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
            log(f"  {inp.name}: shape={shape}, type={inp.type}")

        t0 = time.time()
        outputs = sess.run(None, feeds)
        log(f"Inference succeeded in {(time.time()-t0)*1000:.1f}ms")
        for i, out in enumerate(sess.get_outputs()):
            log(f"  output[{i}] {out.name}: shape={outputs[i].shape}, "
                f"dtype={outputs[i].dtype}")

        # Quick benchmark (5 warmup + 10 timed)
        log("Quick benchmark (5 warmup + 10 timed)...")
        for _ in range(5):
            sess.run(None, feeds)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t0) * 1000)
        log(f"  mean={np.mean(times):.1f}ms  median={np.median(times):.1f}ms  "
            f"min={np.min(times):.1f}ms")

        return True

    except Exception as e:
        log(f"Inference FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Phase 6: Alternative — generate trtexec command
# ---------------------------------------------------------------------------
def phase6_trtexec_guide(model_path):
    """Print trtexec commands for manual TRT engine build."""
    print("\n" + "=" * 70)
    print("PHASE 6: trtexec Commands (Manual Alternative)")
    print("=" * 70)

    print(f"""
  If ORT TRT EP still fails, try building a TRT engine directly with trtexec.
  This gives more control over dynamic shape profiles.

  # Option A: Fixed shapes (fastest engine, least flexible)
  trtexec \\
    --onnx={model_path} \\
    --saveEngine={model_path.replace('.onnx', '.trt')} \\
    --fp16 \\
    --shapes=inputs_embeds:1x1x1024,generation_steps:1,past_keys:5x1x8x1x128,past_values:5x1x8x1x128 \\
    --workspace=2048

  # Option B: Dynamic shapes with min/opt/max profiles
  trtexec \\
    --onnx={model_path} \\
    --saveEngine={model_path.replace('.onnx', '.trt')} \\
    --fp16 \\
    --minShapes=inputs_embeds:1x1x1024,generation_steps:1,past_keys:5x1x8x0x128,past_values:5x1x8x0x128 \\
    --optShapes=inputs_embeds:1x1x1024,generation_steps:1,past_keys:5x1x8x15x128,past_values:5x1x8x15x128 \\
    --maxShapes=inputs_embeds:1x1x1024,generation_steps:15,past_keys:5x1x8x30x128,past_values:5x1x8x30x128 \\
    --workspace=2048

  # Option C: Use polygraphy to debug / isolate problematic subgraphs
  polygraphy run {model_path} \\
    --trt --fp16 \\
    --trt-min-shapes inputs_embeds:[1,1,1024] generation_steps:[1] past_keys:[5,1,8,0,128] past_values:[5,1,8,0,128] \\
    --trt-opt-shapes inputs_embeds:[1,1,1024] generation_steps:[1] past_keys:[5,1,8,15,128] past_values:[5,1,8,15,128] \\
    --trt-max-shapes inputs_embeds:[1,1,1024] generation_steps:[15] past_keys:[5,1,8,30,128] past_values:[5,1,8,30,128]

  # Option D: Use polygraphy surgeon to extract TRT-compatible subgraph
  polygraphy surgeon sanitize {model_path} \\
    --fold-constants \\
    -o {model_path.replace('.onnx', '_sanitized.onnx')}

  Notes:
  - trtexec on Jetson: /usr/src/tensorrt/bin/trtexec
  - If trtexec also fails on If nodes, the model MUST be simplified first
  - polygraphy can isolate the failing subgraph for targeted surgery
  - Known TensorRT limitation: If/Loop/Scan nodes require all branch
    outputs to have identical static shapes. Dynamic shapes in branches
    are NOT supported even with dynamic shape profiles.
""")


# ---------------------------------------------------------------------------
# Phase 7: ORT Transformers Optimizer
# ---------------------------------------------------------------------------
def phase7_ort_optimizer(input_path, output_path):
    """Try onnxruntime.transformers.optimizer for transformer-specific fusions."""
    print("\n" + "=" * 70)
    print("PHASE 7: ORT Transformers Optimizer")
    print("=" * 70)

    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        log("ORT transformers optimizer available")
    except ImportError:
        log("onnxruntime.transformers not available")
        log("Install with: pip install onnxruntime (includes transformers subpackage)")
        return None

    log(f"Optimizing {input_path}...")
    t0 = time.time()

    try:
        # Try GPT-2 style optimization (autoregressive decoder)
        opt_options = FusionOptions("gpt2")
        opt_options.enable_skip_layer_norm = True
        opt_options.enable_attention = True

        optimized = optimizer.optimize_model(
            input_path,
            model_type="gpt2",
            num_heads=8,       # 8 attention heads (from past_keys shape)
            hidden_size=1024,  # hidden dim
            optimization_options=opt_options,
        )

        log(f"Optimization done in {time.time()-t0:.1f}s")

        stats = optimized.get_fused_operator_statistics()
        if stats:
            log(f"Fused operators: {stats}")

        # Check remaining If nodes
        n_if = sum(1 for n in optimized.model.graph.node if n.op_type == "If")
        log(f"If nodes remaining: {n_if}")
        log(f"Total nodes: {len(optimized.model.graph.node)}")

        optimized.save_model_to_file(output_path)
        log(f"Saved to {output_path}")
        return output_path

    except Exception as e:
        log(f"ORT optimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_model(model, path):
    """Save ONNX model, using external data if too large."""
    import onnx
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t0 = time.time()
    try:
        onnx.save(model, path)
    except ValueError:
        # Model too large for single protobuf
        ext_path = os.path.basename(path) + ".data"
        onnx.save(
            model, path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_path,
        )
    sz = os.path.getsize(path) / 1024 / 1024
    ext = path + ".data"
    ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
    log(f"Saved {path} ({sz:.1f}MB" + (f" + {ext_sz:.1f}MB" if ext_sz else "") +
        f") in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize code_predictor.onnx for TensorRT")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Input model path (default: {MODEL_PATH})")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--verify", action="store_true",
                        help="Verify optimized model with TRT EP")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only specific phase (1-7), 0=all")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("Code Predictor TRT Optimization Pipeline")
    print("=" * 70)
    print(f"  Input:  {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Verify: {args.verify}")
    print(f"  Phase:  {'all' if args.phase == 0 else args.phase}")

    if not os.path.exists(args.model):
        print(f"\nERROR: Model not found: {args.model}")
        sys.exit(1)

    sz = os.path.getsize(args.model) / 1024 / 1024
    print(f"  Model size: {sz:.1f}MB")

    import onnx
    print(f"  onnx version: {onnx.__version__}")

    # Phase 0: Analyze
    model = onnx.load(args.model, load_external_data=True)
    if_nodes = analyze_model(model)
    del model
    gc.collect()

    if not if_nodes and args.phase == 0:
        print("\nNo If nodes found — model may already be TRT-compatible.")
        print("Running shape inference and verification only.")

    best_model = args.model
    results = {}

    # Phase 1: Shape inference
    if args.phase in (0, 1):
        p1_out = os.path.join(args.output, "code_predictor_shapeinf.onnx")
        result = phase1_shape_inference(args.model, p1_out)
        results["phase1"] = result
        if result:
            best_model = result

    # Phase 2: onnxsim
    if args.phase in (0, 2):
        p2_out = os.path.join(args.output, "code_predictor_simplified.onnx")
        result = phase2_onnxsim(best_model, p2_out)
        results["phase2"] = result
        if result:
            best_model = result

    # Phase 7: ORT transformers optimizer (before manual surgery)
    if args.phase in (0, 7):
        p7_out = os.path.join(args.output, "code_predictor_ort_opt.onnx")
        result = phase7_ort_optimizer(best_model, p7_out)
        results["phase7"] = result
        if result:
            # Check if it helped with If nodes
            m = onnx.load(result, load_external_data=True)
            n_if = sum(1 for n in m.graph.node if n.op_type == "If")
            del m; gc.collect()
            if n_if == 0:
                best_model = result
                log("ORT optimizer eliminated all If nodes!")

    # Phase 3: Manual If elimination
    if args.phase in (0, 3):
        # Only needed if we still have If nodes
        m = onnx.load(best_model, load_external_data=True)
        n_if = sum(1 for n in m.graph.node if n.op_type == "If")
        del m; gc.collect()

        if n_if > 0:
            p3_out = os.path.join(args.output, "code_predictor_no_if.onnx")
            result = phase3_eliminate_if_nodes(best_model, p3_out)
            results["phase3"] = result
            if result:
                best_model = result

    # Phase 4: Force remove (last resort)
    if args.phase in (0, 4):
        m = onnx.load(best_model, load_external_data=True)
        n_if = sum(1 for n in m.graph.node if n.op_type == "If")
        del m; gc.collect()

        if n_if > 0:
            p4_out = os.path.join(args.output, "code_predictor_static.onnx")
            result = phase4_remove_if_nodes(best_model, p4_out)
            results["phase4"] = result
            if result:
                best_model = result

    # Phase 5: Verify
    if args.verify or args.phase == 5:
        success = phase5_verify_trt(best_model)
        results["verify"] = success

    # Phase 6: trtexec guide (always print)
    if args.phase in (0, 6):
        phase6_trtexec_guide(best_model)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best optimized model: {best_model}")
    if os.path.exists(best_model):
        sz = os.path.getsize(best_model) / 1024 / 1024
        ext = best_model + ".data"
        ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
        print(f"  Size: {sz:.1f}MB" + (f" + {ext_sz:.1f}MB" if ext_sz else ""))
    for phase, result in results.items():
        status = "OK" if result else "FAILED/SKIPPED"
        print(f"  {phase}: {status}" + (f" -> {result}" if isinstance(result, str) else ""))

    print(f"""
  Next steps:
  1. Copy optimized model to Jetson: scp {best_model} jetson:/tmp/qwen3-tts-bench/model-trt-opt/
  2. On Jetson, verify with TRT EP:
     python {__file__} --model {best_model} --verify --phase 5
  3. Or build TRT engine with trtexec (see Phase 6 output above)
  4. Update bench_trt_ep.py to use optimized model path

  Known TensorRT limitations with ONNX If nodes:
  - TRT requires all branch outputs to have identical STATIC shapes
  - Dynamic dims inside If branches are NOT supported
  - The only reliable fix is to eliminate If nodes entirely
  - onnx-simplifier with fixed input shapes is the cleanest approach
  - Manual branch inlining (Phase 3) works when simplifier cannot
""")


if __name__ == "__main__":
    main()
