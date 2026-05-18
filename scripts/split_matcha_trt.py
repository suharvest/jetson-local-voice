#!/usr/bin/env python3
"""Split Matcha acoustic ONNX for Jetson TensorRT.

The full Matcha acoustic graph computes the mel length from the duration
predictor, then uses that value in shape-producing subgraphs. TensorRT 10.3
cannot compile that data-dependent shape chain reliably. This split keeps the
dynamic duration/length part in a small ORT encoder graph and moves the heavy
ODE estimator block into a fixed-shape TensorRT-friendly graph.

Outputs:
  matcha_encoder_trt.onnx        x -> mu, mask, z0
  matcha_estimator_step{0,1,2}_trt.onnx
                                  z, mu, mask -> velocity for each ODE step
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.utils import Extractor

MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3

BATCH = 1
MEL_BINS = 80
MAX_FRAMES = 600
FRAME_DIM = "T"

MU_NAME = "/Transpose_3_output_0"
MASK_NAME = "/Cast_3_output_0"
Z0_NAME = "/decoder/Mul_output_0"


def _shape(vi: onnx.ValueInfoProto) -> list[int | str]:
    return [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]


def split_onnx(onnx_path: Path, output_dir: Path) -> None:
    model = onnx.load(str(onnx_path))
    model = onnx.shape_inference.infer_shapes(model)
    extractor = Extractor(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc_model = extractor.extract_model(
        input_names=["x", "x_length", "noise_scale", "length_scale"],
        output_names=[MU_NAME, MASK_NAME, Z0_NAME],
    )
    enc_path = output_dir / "matcha_encoder_trt.onnx"
    onnx.save(enc_model, str(enc_path))

    estimator_specs = [
        (0, Z0_NAME, "/decoder/estimator/Mul_5_output_0"),
        (1, "/decoder/Add_output_0", "/decoder/estimator_1/Mul_5_output_0"),
        (2, "/decoder/Add_1_output_0", "/decoder/estimator_2/Mul_5_output_0"),
    ]
    est_models = []
    for step, z_name, velocity_name in estimator_specs:
        est_model = _extract_estimator_step(extractor, step, z_name, velocity_name)
        est_path = output_dir / f"matcha_estimator_step{step}_trt.onnx"
        onnx.save(est_model, str(est_path))
        est_models.append((step, est_path, est_model))

    print(f"encoder:   {enc_path} ({len(enc_model.graph.node)} nodes)")
    for item in enc_model.graph.output:
        print(f"  out {item.name}: {_shape(item)}")
    for step, est_path, est_model in est_models:
        print(f"estimator[{step}]: {est_path} ({len(est_model.graph.node)} nodes)")
        for item in est_model.graph.input:
            print(f"  in  {item.name}: {_shape(item)}")
        for item in est_model.graph.output:
            print(f"  out {item.name}: {_shape(item)}")


def _extract_estimator_step(
    extractor: Extractor,
    step: int,
    z_name: str,
    velocity_name: str,
) -> onnx.ModelProto:
    model = extractor.extract_model(
        input_names=[z_name, MU_NAME, MASK_NAME],
        output_names=[velocity_name],
    )
    rename_map = {
        z_name: "z",
        MU_NAME: "mu",
        MASK_NAME: "mask",
        velocity_name: "velocity",
    }
    for node in model.graph.node:
        for idx, value in enumerate(node.input):
            if value in rename_map:
                node.input[idx] = rename_map[value]
        for idx, value in enumerate(node.output):
            if value in rename_map:
                node.output[idx] = rename_map[value]

    renamed_values = set(rename_map) | set(rename_map.values())
    value_info = [vi for vi in model.graph.value_info if vi.name not in renamed_values]
    graph = helper.make_graph(
        list(model.graph.node),
        f"matcha-estimator-step{step}-trt",
        [
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [BATCH, MEL_BINS, FRAME_DIM]),
            helper.make_tensor_value_info("mu", TensorProto.FLOAT, [BATCH, MEL_BINS, FRAME_DIM]),
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, [BATCH, 1, FRAME_DIM]),
        ],
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT, [BATCH, MEL_BINS, FRAME_DIM])],
        initializer=list(model.graph.initializer),
        value_info=value_info,
    )
    out = helper.make_model(graph, opset_imports=model.opset_import)
    out.ir_version = model.ir_version
    return out


def verify_split(onnx_path: Path, output_dir: Path, noise_scale_value: float = 0.0) -> bool:
    import onnxruntime as ort

    x = np.zeros((1, 80), dtype=np.int64)
    x[0, :10] = np.arange(1, 11, dtype=np.int64)
    x_length = np.array([10], dtype=np.int64)
    # Full ONNX and split encoder each own a RandomNormalLike node. With
    # non-zero noise they draw different z0 samples, so direct full-vs-split
    # diff is not a valid parity check. noise_scale=0 makes z0 deterministic
    # while still exercising the same duration, estimator, and denorm graph.
    noise_scale = np.array([noise_scale_value], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)

    full_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    full_mel = full_sess.run(
        None,
        {"x": x, "x_length": x_length, "noise_scale": noise_scale, "length_scale": length_scale},
    )[0]

    enc_sess = ort.InferenceSession(str(output_dir / "matcha_encoder_trt.onnx"), providers=["CPUExecutionProvider"])
    mu, mask, z = enc_sess.run(
        None,
        {"x": x, "x_length": x_length, "noise_scale": noise_scale, "length_scale": length_scale},
    )
    mu = np.ascontiguousarray(mu.astype(np.float32))
    mask = np.ascontiguousarray(mask.astype(np.float32))
    z = np.ascontiguousarray(z.astype(np.float32))

    est_sessions = [
        ort.InferenceSession(
            str(output_dir / f"matcha_estimator_step{i}_trt.onnx"),
            providers=["CPUExecutionProvider"],
        )
        for i in range(N_ODE_STEPS)
    ]
    for step in range(N_ODE_STEPS):
        feeds = {"z": z, "mu": mu, "mask": mask}
        velocity = est_sessions[step].run(None, feeds)[0]
        z = z + ODE_DT * velocity

    frames = full_mel.shape[2]
    split_mel = z[:, :, :frames] * MEL_SIGMA + MEL_MEAN
    diff = np.abs(full_mel - split_mel)
    print(f"verify: full={full_mel.shape} split={split_mel.shape} noise_scale={noise_scale_value}")
    print(f"  max_abs={diff.max():.6g} mean_abs={diff.mean():.6g}")
    return bool(diff.max() < 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--onnx", dest="input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-noise-scale",
        type=float,
        default=0.0,
        help="Use 0 for deterministic full-vs-split parity; non-zero randomizes z0 in both sessions independently.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.input)
    output_dir = Path(args.output_dir)
    split_onnx(onnx_path, output_dir)
    if args.verify and not verify_split(onnx_path, output_dir, args.verify_noise_scale):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
