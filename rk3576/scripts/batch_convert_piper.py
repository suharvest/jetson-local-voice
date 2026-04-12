#!/usr/bin/env python3
"""Batch download and convert Piper TTS models to RKNN.

Downloads Piper voice models from HuggingFace and runs the full ONNX surgery
pipeline (fix_piper_rknn.py) before exporting to .rknn format.

Usage:
  python batch_convert_piper.py --target rk3588 --output-dir /tmp/piper-rknn-models
  python batch_convert_piper.py --target rk3576 --languages en_US,zh_CN,ja_JP
  python batch_convert_piper.py --target rk3588 --languages all
"""

import os
import sys
import json
import argparse
import tempfile
import urllib.request
import urllib.error
import time
import traceback

# ---------------------------------------------------------------------------
# Model registry: language code → (hf_path, model_name)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- Existing / already deployed ---
    'en_US':  ('en/en_US/lessac/medium',      'en_US-lessac-medium'),
    'zh_CN':  ('zh/zh_CN/huayan/medium',      'zh_CN-huayan-medium'),
    'ja_JP':  ('ja/ja_JP/kokoro/medium',      'ja_JP-kokoro-medium'),
    'de_DE':  ('de/de_DE/thorsten/medium',    'de_DE-thorsten-medium'),
    'fr_FR':  ('fr/fr_FR/siwis/medium',       'fr_FR-siwis-medium'),

    # --- Priority expansion languages ---
    'es_ES':  ('es/es_ES/davefx/medium',      'es_ES-davefx-medium'),
    'es_MX':  ('es/es_MX/claude/high',        'es_MX-claude-high'),
    'it_IT':  ('it/it_IT/riccardo/x_low',     'it_IT-riccardo-x_low'),
    'ru_RU':  ('ru/ru_RU/irina/medium',       'ru_RU-irina-medium'),
    'pt_BR':  ('pt/pt_BR/faber/medium',       'pt_BR-faber-medium'),
    'nl_NL':  ('nl/nl_NL/mls_5809/low',       'nl_NL-mls_5809-low'),
    'pl_PL':  ('pl/pl_PL/darkman/medium',     'pl_PL-darkman-medium'),
    'ar_JO':  ('ar/ar_JO/kareem/medium',      'ar_JO-kareem-medium'),
    'tr_TR':  ('tr/tr_TR/dfki/medium',        'tr_TR-dfki-medium'),
    'vi_VN':  ('vi/vi_VN/vivos/x_low',        'vi_VN-vivos-x_low'),
    # ko_KR: not available in Piper — skipped intentionally
    'uk_UA':  ('uk/uk_UA/lada/x_low',         'uk_UA-lada-x_low'),
    'sv_SE':  ('sv/sv_SE/nst/medium',         'sv_SE-nst-medium'),
    'cs_CZ':  ('cs/cs_CZ/jirka/medium',       'cs_CZ-jirka-medium'),
}

HF_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0'


def download_file(url: str, dest: str, max_retries: int = 3) -> bool:
    """Download a file from URL to dest path. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Downloading: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            with open(dest, 'wb') as f:
                f.write(data)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"    Saved: {dest} ({size_mb:.1f} MB)")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    ERROR: 404 Not Found — {url}")
                return False
            print(f"    HTTP error {e.code} (attempt {attempt}/{max_retries}): {e}")
        except Exception as e:
            print(f"    Download error (attempt {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            time.sleep(5 * attempt)
    return False


def apply_onnx_pipeline(onnx_path: str, fixed_path: str, seq_len: int) -> bool:
    """Run the fix_piper_rknn.py ONNX surgery pipeline. Returns True on success."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    # Add fix_piper_rknn.py to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from fix_piper_rknn import (
        load_and_simplify,
        fix_range_nodes,
        fix_erf_nodes,
        fix_softplus_nodes,
        fix_ceil_ops,
        fix_nonzero_nodes,
        replace_cpu_fallback_ops,
        fix_3d_matmul_for_rknn,
        fix_random_noise,
    )

    tokens = np.zeros((1, seq_len), dtype=np.int64)
    tokens[0, :5] = [1, 2, 3, 4, 5]
    test_inputs = {
        'input': tokens,
        'input_lengths': np.array([5], dtype=np.int64),
        'scales': np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }

    # Auto-detect multi-speaker models: add 'sid' if the model requires it
    _probe = onnx.load(onnx_path)
    _input_names = {inp.name for inp in _probe.graph.input}
    if 'sid' in _input_names:
        print("    [auto] Multi-speaker model detected — adding sid=0 to test inputs")
        test_inputs['sid'] = np.array([0], dtype=np.int64)
    del _probe

    print(f"    [1/6] onnxsim simplification (seq_len={seq_len})...")
    model = load_and_simplify(onnx_path, seq_len)

    print("    [2/6] Replacing Range nodes...")
    model = fix_range_nodes(model, test_inputs)

    print("    [3/6] Replacing Erf nodes...")
    model = fix_erf_nodes(model)

    print("    [4/6] Replacing Softplus nodes...")
    model = fix_softplus_nodes(model)

    print("    [4b] Replacing Ceil ops...")
    model = fix_ceil_ops(model)

    print("    [5a/6] Replacing NonZero nodes...")
    model = fix_nonzero_nodes(model, test_inputs)

    # Re-run onnxsim after baking NonZero constants
    print("    [5a-sim] Re-running onnxsim after NonZero baking...")
    import onnxsim
    tmp_nz = tempfile.mktemp(suffix='.onnx')
    onnx.save(model, tmp_nz)
    model2, ok2 = onnxsim.simplify(onnx.load(tmp_nz))
    os.unlink(tmp_nz)
    print(f"      Re-simplified: ok={ok2}, nodes={len(model2.graph.node)}")
    model = model2

    print("    [5a2/6] Replacing ScatterND/GatherND/CumSum with NPU-native ops...")
    model = replace_cpu_fallback_ops(model, test_inputs)

    print("    [5b/6] Fixing 3D MatMul for RKNN exMatMul bug...")
    model = fix_3d_matmul_for_rknn(model, test_inputs)

    print("    [5c/6] Replacing RandomNormalLike / RandomUniformLike...")
    model = fix_random_noise(model, test_inputs)

    print("    [verify] ORT verification...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"      Warning (shape inference): {e}")

    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    out = sess.run(None, test_inputs)
    audio = out[0]
    import numpy as np
    print(f"      Output shape: {audio.shape}, RMS={float(np.sqrt(np.mean(audio**2))):.4f}")
    print("      ORT OK")

    onnx.save(model, fixed_path)
    sz = os.path.getsize(fixed_path) / (1024 * 1024)
    print(f"      Saved fixed ONNX: {fixed_path} ({sz:.1f} MB)")
    return True


def build_rknn(fixed_onnx: str, rknn_path: str, target: str) -> bool:
    """Run RKNN build and export. Returns True on success."""
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    ret = rknn.config(target_platform=target, optimization_level=0)
    if ret != 0:
        print(f"    RKNN config failed: {ret}")
        return False

    ret = rknn.load_onnx(model=fixed_onnx)
    if ret != 0:
        print(f"    RKNN load_onnx failed: {ret}")
        return False

    print(f"    RKNN build (target={target})...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"    RKNN build failed: {ret}")
        return False

    ret = rknn.export_rknn(rknn_path)
    rknn.release()
    if ret != 0:
        print(f"    RKNN export failed: {ret}")
        return False

    sz = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"    RKNN exported: {rknn_path} ({sz:.1f} MB)")
    return True


def convert_language(lang: str, target: str, output_dir: str, seq_len: int,
                     keep_intermediate: bool = False) -> dict:
    """Download, fix, and convert a single language model.

    Returns a result dict with keys: lang, status, rknn_path, config_path, error.
    """
    result = {'lang': lang, 'status': 'fail', 'rknn_path': None, 'config_path': None, 'error': None}

    if lang not in MODEL_REGISTRY:
        result['error'] = f"Unknown language: {lang}"
        return result

    hf_subpath, model_name = MODEL_REGISTRY[lang]
    onnx_url  = f"{HF_BASE}/{hf_subpath}/{model_name}.onnx"
    config_url = f"{HF_BASE}/{hf_subpath}/{model_name}.onnx.json"

    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    raw_onnx   = os.path.join(lang_dir, f"{model_name}.onnx")
    fixed_onnx = os.path.join(lang_dir, f"{model_name}-rknn-ready.onnx")
    rknn_path  = os.path.join(lang_dir, f"{model_name}.rknn")
    config_path = os.path.join(lang_dir, f"{model_name}.onnx.json")

    # --- Download ONNX ---
    print(f"\n  [{lang}] Downloading ONNX model...")
    if not download_file(onnx_url, raw_onnx):
        result['error'] = f"Download failed: {onnx_url}"
        return result

    # --- Download config JSON ---
    print(f"  [{lang}] Downloading config JSON...")
    if not download_file(config_url, config_path):
        print(f"  [{lang}] WARNING: config JSON download failed (non-fatal)")
    else:
        result['config_path'] = config_path

    # --- ONNX surgery ---
    print(f"  [{lang}] Running ONNX fix pipeline...")
    try:
        ok = apply_onnx_pipeline(raw_onnx, fixed_onnx, seq_len)
        if not ok:
            result['error'] = "ONNX pipeline returned False"
            return result
    except Exception as e:
        result['error'] = f"ONNX pipeline exception: {e}\n{traceback.format_exc()}"
        return result

    # --- RKNN build ---
    print(f"  [{lang}] Building RKNN ({target})...")
    try:
        ok = build_rknn(fixed_onnx, rknn_path, target)
        if not ok:
            result['error'] = "RKNN build returned False"
            return result
    except ImportError:
        # RKNN toolkit not installed — save fixed ONNX and skip
        print(f"  [{lang}] WARNING: rknn.api not available, skipping RKNN build. Fixed ONNX saved.")
        result['status'] = 'onnx_only'
        result['rknn_path'] = fixed_onnx
        if not keep_intermediate:
            # keep raw onnx only if rknn failed
            pass
        return result
    except Exception as e:
        result['error'] = f"RKNN build exception: {e}\n{traceback.format_exc()}"
        return result

    # --- Cleanup intermediate files ---
    if not keep_intermediate:
        for f in [raw_onnx, fixed_onnx]:
            try:
                os.remove(f)
            except OSError:
                pass

    result['status'] = 'success'
    result['rknn_path'] = rknn_path
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch download and convert Piper TTS models to RKNN'
    )
    parser.add_argument(
        '--target', default='rk3588',
        choices=['rk3576', 'rk3588', 'rk3562', 'rv1103', 'rv1106'],
        help='RKNN target platform (default: rk3588)'
    )
    parser.add_argument(
        '--output-dir', default='/tmp/piper-rknn-models',
        help='Directory to write converted models (default: /tmp/piper-rknn-models)'
    )
    parser.add_argument(
        '--languages', default='all',
        help='Comma-separated language codes, or "all" (default: all). '
             f'Available: {", ".join(sorted(MODEL_REGISTRY.keys()))}'
    )
    parser.add_argument(
        '--seq-len', type=int, default=128,
        help='Token sequence length / bucket size for ONNX surgery (default: 128)'
    )
    parser.add_argument(
        '--keep-intermediate', action='store_true',
        help='Keep raw and fixed ONNX files after RKNN conversion'
    )
    args = parser.parse_args()

    # Resolve language list
    if args.languages.strip().lower() == 'all':
        langs = sorted(MODEL_REGISTRY.keys())
    else:
        langs = [l.strip() for l in args.languages.split(',') if l.strip()]

    print("=" * 60)
    print("Piper TTS → RKNN Batch Converter")
    print(f"  Target:     {args.target}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Languages:  {', '.join(langs)}")
    print(f"  Seq len:    {args.seq_len}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for lang in langs:
        print(f"\n{'='*60}")
        print(f"Processing: {lang}")
        print('='*60)
        t0 = time.time()
        result = convert_language(
            lang=lang,
            target=args.target,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            keep_intermediate=args.keep_intermediate,
        )
        elapsed = time.time() - t0
        result['elapsed_s'] = round(elapsed, 1)
        results.append(result)

        if result['status'] == 'success':
            rknn_sz = os.path.getsize(result['rknn_path']) / (1024 * 1024) if result['rknn_path'] else 0
            print(f"\n  [{lang}] SUCCESS in {elapsed:.0f}s — {result['rknn_path']} ({rknn_sz:.1f} MB)")
        elif result['status'] == 'onnx_only':
            print(f"\n  [{lang}] ONNX ONLY (no RKNN toolkit) in {elapsed:.0f}s — {result['rknn_path']}")
        else:
            print(f"\n  [{lang}] FAILED in {elapsed:.0f}s — {result['error']}")

    # Save results JSON
    results_file = os.path.join(args.output_dir, 'conversion_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    succeeded = [r for r in results if r['status'] == 'success']
    onnx_only = [r for r in results if r['status'] == 'onnx_only']
    failed    = [r for r in results if r['status'] == 'fail']

    for r in succeeded:
        sz = os.path.getsize(r['rknn_path']) / (1024 * 1024) if r['rknn_path'] else 0
        print(f"  OK   {r['lang']:12s}  {r['elapsed_s']:5.0f}s  {sz:5.1f} MB  {r['rknn_path']}")
    for r in onnx_only:
        print(f"  ONNX {r['lang']:12s}  {r['elapsed_s']:5.0f}s  (no rknn toolkit)  {r['rknn_path']}")
    for r in failed:
        print(f"  FAIL {r['lang']:12s}  {r['elapsed_s']:5.0f}s  {r['error'][:80]}")

    print(f"\nResults: {len(succeeded)} succeeded, {len(onnx_only)} onnx-only, {len(failed)} failed")
    print(f"Results JSON: {results_file}")

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
