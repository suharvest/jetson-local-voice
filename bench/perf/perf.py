#!/usr/bin/env python3
"""Seeed Local Voice — unified perf harness.

Five scenarios, all output one JSON + one Markdown report under results/:
  perf.py asr        --base-url ... --runs 10 --warmup 3
  perf.py tts        --base-url ... --runs 10 --warmup 3
  perf.py v2v        --base-url ... --llm-delay 0   --eos vad
  perf.py v2v        --base-url ... --llm-delay 800 --eos vad
  perf.py concurrent --base-url ... --parallel 2 --mode asr_tts_simul
  perf.py matrix     --base-url ...            # runs all of the above
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import ASRClient, TTSClient
from runners import (load_corpus, load_prompts,
                     run_asr, run_tts, run_v2v_bench, run_concurrent,
                     run_asr_noisy)
from stats import summarize, save_results
from memory import MemorySampler
from cold_start import measure_container_boot
from clone_bench import run_clone
from stability import run_stability

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _common_meta(args, scenario: str) -> dict:
    import platform, socket
    mode = getattr(args, "mode_label", None) or _infer_mode(getattr(args, "base_url", ""))
    return {
        "scenario": scenario,
        "mode": mode,                          # 'local' = on-device loopback, 'remote' = cross-network
        "base_url": args.base_url,
        "warmup": getattr(args, "warmup", None),
        "runs":   getattr(args, "runs", None),
        "container": getattr(args, "container", None) or "(none)",
        "client_host": socket.gethostname(),    # who ran the client (for remote-mode attribution)
        "client_platform": platform.platform(),
    }


def _infer_mode(base_url: str) -> str:
    """Default mode inference: loopback URL → local; everything else → remote."""
    if not base_url:
        return "unknown"
    return "local" if ("localhost" in base_url or "127.0.0.1" in base_url) else "remote"


def _scenario_tag(args, scenario: str) -> str:
    """Filename-friendly tag including mode."""
    mode = getattr(args, "mode_label", None) or _infer_mode(getattr(args, "base_url", ""))
    return f"{scenario}_{mode}"


def cmd_asr(args):
    asr = ASRClient(
        args.base_url,
        chunk_ms=args.chunk_ms,
        realtime=args.realtime,
        vad_backend=args.vad_backend,
        vad_silence_ms=args.vad_silence_ms,
    )
    corpus = load_corpus(category=args.category, lang=args.lang)
    if not corpus:
        sys.exit("No corpus files found; run corpus/fetch.py first.")
    print(f"Corpus: {len(corpus)} files (category={args.category or 'all'}, lang={args.lang or 'all'})")
    with MemorySampler(args.container) as mem:
        rows = run_asr(asr, corpus, warmup=args.warmup, runs=args.runs,
                       mode=args.mode, eos_mode=args.eos)
    summary = summarize(rows, group_by=("category", "lang"),
                        metrics=("rtf", "finalize_rtf", "tfd_ms", "error_rate",
                                 "eos_to_final_ms", "processing_ms"))
    meta = {**_common_meta(args, "asr"), "mode": args.mode, "eos": args.eos,
            "chunk_ms": args.chunk_ms, "vad_backend": args.vad_backend,
            "vad_silence_ms": args.vad_silence_ms}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, f"asr_{args.mode}"), rows, summary,
                          mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_tts(args):
    tts = TTSClient(args.base_url, stream=not args.no_stream, voice=args.voice)
    prompts = load_prompts(category=args.category, lang=args.lang)
    print(f"Prompts: {len(prompts)} (category={args.category or 'all'}, lang={args.lang or 'all'})")
    with MemorySampler(args.container) as mem:
        rows = run_tts(tts, prompts, warmup=args.warmup, runs=args.runs)
    summary = summarize(rows, group_by=("category", "lang"),
                        metrics=("rtf", "tfd_ms", "total_ms"))
    meta = {**_common_meta(args, "tts"), "stream": not args.no_stream,
            "voice": args.voice or "(default)"}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, "tts"), rows, summary, mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_v2v(args):
    asr = ASRClient(
        args.base_url,
        chunk_ms=args.chunk_ms,
        realtime=args.realtime,
        vad_backend=args.vad_backend,
        vad_silence_ms=args.vad_silence_ms,
    )
    tts = TTSClient(args.base_url, stream=True, voice=args.voice)
    corpus = load_corpus(category=args.category, lang=args.lang)
    if not corpus:
        sys.exit("No corpus files found; run corpus/fetch.py first.")
    print(f"V2V: {len(corpus)} files, eos={args.eos}, llm_delay={args.llm_delay}ms")
    with MemorySampler(args.container) as mem:
        rows = run_v2v_bench(asr, tts, corpus, warmup=args.warmup, runs=args.runs,
                             eos_mode=args.eos, llm_delay_ms=args.llm_delay)
    summary = summarize(rows, group_by=("category", "lang"),
                        metrics=("eos_to_first_audio_ms", "asr_finalize_ms",
                                 "tts_tfd_ms", "tts_total_ms"))
    meta = {**_common_meta(args, "v2v"), "eos": args.eos,
            "llm_delay_ms": args.llm_delay, "vad_backend": args.vad_backend,
            "vad_silence_ms": args.vad_silence_ms}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, f"v2v_{args.eos}_llm{int(args.llm_delay)}"),
                          rows, summary, mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_concurrent(args):
    asr = ASRClient(args.base_url)
    tts = TTSClient(args.base_url, stream=True, voice=args.voice)
    corpus = load_corpus(category=args.category)
    prompts = load_prompts(category=args.category)
    if not corpus and args.mode != "tts_only":
        sys.exit("Concurrent ASR needs corpus.")
    print(f"Concurrent: parallel={args.parallel}, mode={args.mode}, "
          f"runs_per_worker={args.runs}")
    with MemorySampler(args.container) as mem:
        rows = run_concurrent(asr, tts, corpus, prompts,
                              parallel=args.parallel,
                              runs_per_worker=args.runs,
                              mode=args.mode)
    summary = summarize(rows, group_by=("kind",),
                        metrics=("rtf", "tfd_ms", "wall_ms",
                                 "eos_to_final_ms", "total_ms"))
    meta = {**_common_meta(args, "concurrent"),
            "parallel": args.parallel, "mode": args.mode}
    jp, mp = save_results(RESULTS_DIR,
                          _scenario_tag(args, f"concurrent_{args.mode}_p{args.parallel}"),
                          rows, summary, mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_noise(args):
    asr = ASRClient(args.base_url)
    corpus = load_corpus(category=args.category, lang=args.lang)
    if not corpus:
        sys.exit("No corpus files; run corpus/fetch.py first.")
    all_rows = []
    for snr in args.snr:
        print(f"\n--- SNR {snr} dB ({args.noise_type}) ---")
        rows = run_asr_noisy(asr, corpus, snr_db=snr, noise_type=args.noise_type,
                             warmup=args.warmup, runs=args.runs)
        all_rows.extend(rows)
    summary = summarize(all_rows, group_by=("snr_db", "category", "lang"),
                        metrics=("rtf", "error_rate", "tfd_ms"))
    meta = {**_common_meta(args, "noise"),
            "noise_type": args.noise_type, "snr_list": args.snr}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, f"noise_{args.noise_type}"),
                          all_rows, summary, None, meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_stability(args):
    asr = ASRClient(args.base_url)
    tts = TTSClient(args.base_url, stream=True)
    corpus  = load_corpus()
    prompts = load_prompts()
    print(f"Stability: mode={args.mode}, duration={args.duration_min}min, "
          f"container={args.container or '(none)'}")
    with MemorySampler(args.container) as mem:
        out = run_stability(asr, tts, corpus, prompts,
                            duration_s=args.duration_min * 60,
                            mode=args.mode, memory_sampler=mem)
    out["memory_summary"] = mem.summary()
    print("\n=== Drift ===")
    print(f"  early RTF p50 = {out['drift']['early_rtf_p50']}  (n={out['drift']['early_n']})")
    print(f"  late  RTF p50 = {out['drift']['late_rtf_p50']}   (n={out['drift']['late_n']})")
    print(f"  drift = {out['drift']['rtf_drift_pct']}%")
    if out["memory_drift"]:
        md = out["memory_drift"]
        print(f"\n  RSS head={md['head_mib_mean']}MiB  tail={md['tail_mib_mean']}MiB  "
              f"growth={md['rss_growth_mib']}MiB  peak={md['peak_mib']}MiB")
    meta = {**_common_meta(args, "stability"),
            "duration_min": args.duration_min, "mode": args.mode}
    summary = {"drift": out["drift"], "memory_drift": out["memory_drift"]}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, f"stability_{args.mode}"),
                          out["rows"], summary, mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_clone(args):
    refs_dir = Path(args.refs)
    if not refs_dir.is_dir():
        sys.exit(f"--refs is not a directory: {refs_dir}")
    refs = sorted([p for p in refs_dir.iterdir() if p.suffix.lower() == ".wav"])
    if not refs:
        sys.exit(f"no .wav files in {refs_dir}")
    texts = json.loads(Path(args.texts).read_text())["texts"]
    print(f"Clone: {len(refs)} refs, {len(texts)} texts, warmup={args.warmup}, runs={args.runs}")
    with MemorySampler(args.container) as mem:
        rows = run_clone(args.base_url, refs, texts,
                         warmup=args.warmup, runs=args.runs,
                         skip_similarity=args.skip_similarity)
    summary = summarize(rows, group_by=("lang",),
                        metrics=("rtf", "embed_ms", "synth_ms", "total_ms", "similarity"))
    meta = {**_common_meta(args, "clone"),
            "refs_dir": str(refs_dir), "texts_file": str(args.texts),
            "skip_similarity": args.skip_similarity}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, "clone"), rows, summary, mem.summary(), meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_boot(args):
    if not args.container:
        sys.exit("--container is required for boot measurement")
    runs = []
    for i in range(args.runs):
        print(f"\nBoot run {i+1}/{args.runs} (container={args.container})...")
        r = measure_container_boot(
            args.container, args.base_url,
            health_path=args.health_path, timeout_s=args.timeout_s,
        )
        runs.append(r)
        if r.get("boot_ms") is not None:
            print(f"  docker_restart={r['docker_restart_ms']:.0f}ms  "
                  f"first_connect={r.get('first_connect_ms') or 0:.0f}ms  "
                  f"boot={r['boot_ms']:.0f}ms  "
                  f"(after_restart={r['boot_after_restart_ms']:.0f}ms)")
        else:
            print(f"  FAIL: {r.get('error')}")
    summary = summarize(runs, group_by=(), metrics=("boot_ms", "docker_restart_ms",
                                                     "boot_after_restart_ms"))
    meta = {**_common_meta(args, "boot"), "health_path": args.health_path}
    jp, mp = save_results(RESULTS_DIR, _scenario_tag(args, "boot"), runs, summary, None, meta)
    print(f"\nSaved: {jp}\n       {mp}")


def cmd_matrix(args):
    """Run a full sweep: asr-streaming, tts, v2v(vad/0 + vad/800), concurrent {1,2,4}×{asr,tts,simul}."""
    print("=" * 60)
    print("MATRIX: asr streaming")
    print("=" * 60)
    cmd_asr(argparse.Namespace(**vars(args),
        mode="streaming", eos="vad", chunk_ms=250, realtime=True,
        vad_backend="silero", vad_silence_ms=400,
        category=None, lang=None))
    print("\n" + "=" * 60)
    print("MATRIX: tts")
    print("=" * 60)
    cmd_tts(argparse.Namespace(**vars(args),
        no_stream=False, voice=None, category=None, lang=None))
    print("\n" + "=" * 60)
    print("MATRIX: v2v vad, llm=0")
    print("=" * 60)
    cmd_v2v(argparse.Namespace(**vars(args),
        eos="vad", llm_delay=0, voice=None, chunk_ms=250, realtime=True,
        vad_backend="silero", vad_silence_ms=400,
        category=None, lang=None))
    print("\n" + "=" * 60)
    print("MATRIX: v2v vad, llm=800")
    print("=" * 60)
    cmd_v2v(argparse.Namespace(**vars(args),
        eos="vad", llm_delay=800, voice=None, chunk_ms=250, realtime=True,
        vad_backend="silero", vad_silence_ms=400,
        category=None, lang=None))
    for parallel in (1, 2, 4):
        for mode in ("asr_only", "tts_only", "asr_tts_simul"):
            print("\n" + "=" * 60)
            print(f"MATRIX: concurrent parallel={parallel} mode={mode}")
            print("=" * 60)
            cmd_concurrent(argparse.Namespace(**vars(args),
                parallel=parallel, mode=mode, voice=None, category=None))


def cmd_ab(args):
    """Run ASR/V2V A/B sweeps over EOS protocols on the same target."""
    if args.scenario in ("asr", "both"):
        for eos in args.eos_modes:
            print("\n" + "=" * 60)
            print(f"AB: asr eos={eos}")
            print("=" * 60)
            asr_args = vars(args).copy()
            asr_args.update(mode="streaming", eos=eos)
            cmd_asr(argparse.Namespace(**asr_args))
    if args.scenario in ("v2v", "both"):
        for eos in args.eos_modes:
            print("\n" + "=" * 60)
            print(f"AB: v2v eos={eos} llm={args.llm_delay}ms")
            print("=" * 60)
            v2v_args = vars(args).copy()
            v2v_args.update(eos=eos)
            cmd_v2v(argparse.Namespace(**v2v_args))


def main():
    p = argparse.ArgumentParser(prog="perf")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--base-url", default="http://localhost:8000")
        sp.add_argument("--container", default=None,
                        help="docker container name for memory sampling")
        sp.add_argument("--warmup", type=int, default=3)
        sp.add_argument("--runs", type=int, default=10)
        sp.add_argument("--category", choices=["short", "long"], default=None)
        sp.add_argument("--lang", choices=["zh", "en"], default=None)
        sp.add_argument("--mode-label", choices=["local", "remote"], default=None,
                        dest="mode_label",
                        help="override auto-detected client location (loopback=local, else remote). "
                             "Tags result file and meta for cross-device comparison.")

    sp_asr = sub.add_parser("asr"); add_common(sp_asr)
    sp_asr.add_argument("--mode", choices=["streaming", "offline"], default="streaming")
    sp_asr.add_argument("--eos", choices=["forced", "vad", "eou"], default="vad")
    sp_asr.add_argument("--chunk-ms", type=int, default=250)
    sp_asr.add_argument("--vad-backend", choices=["silero", "webrtcvad"], default="silero",
                        help="server-side VAD query param for --eos vad on seeed-local-voice")
    sp_asr.add_argument("--vad-silence-ms", type=int, default=400)
    sp_asr.add_argument("--realtime", action="store_true", default=True)
    sp_asr.add_argument("--no-realtime", dest="realtime", action="store_false")
    sp_asr.set_defaults(func=cmd_asr)

    sp_tts = sub.add_parser("tts"); add_common(sp_tts)
    sp_tts.add_argument("--no-stream", action="store_true")
    sp_tts.add_argument("--voice", default=None)
    sp_tts.set_defaults(func=cmd_tts)

    sp_v2v = sub.add_parser("v2v"); add_common(sp_v2v)
    sp_v2v.add_argument("--eos", choices=["forced", "vad", "eou"], default="vad")
    sp_v2v.add_argument("--llm-delay", type=float, default=0.0,
                        help="LLM placeholder delay in ms (e.g., 800)")
    sp_v2v.add_argument("--voice", default=None)
    sp_v2v.add_argument("--chunk-ms", type=int, default=250)
    sp_v2v.add_argument("--vad-backend", choices=["silero", "webrtcvad"], default="silero",
                        help="server-side VAD query param for --eos vad on seeed-local-voice")
    sp_v2v.add_argument("--vad-silence-ms", type=int, default=400)
    sp_v2v.add_argument("--realtime", action="store_true", default=True)
    sp_v2v.add_argument("--no-realtime", dest="realtime", action="store_false")
    sp_v2v.set_defaults(func=cmd_v2v)

    sp_con = sub.add_parser("concurrent"); add_common(sp_con)
    sp_con.add_argument("--parallel", type=int, default=2, choices=[1, 2, 4])
    sp_con.add_argument("--mode", default="asr_tts_simul",
                        choices=["asr_only", "tts_only", "asr_tts_simul"])
    sp_con.add_argument("--voice", default=None)
    sp_con.set_defaults(func=cmd_concurrent)

    sp_mat = sub.add_parser("matrix"); add_common(sp_mat)
    sp_mat.set_defaults(func=cmd_matrix)

    sp_ab = sub.add_parser("ab"); add_common(sp_ab)
    sp_ab.add_argument("--scenario", choices=["asr", "v2v", "both"], default="both")
    sp_ab.add_argument("--eos-modes", nargs="+", choices=["forced", "eou", "vad"],
                       default=["vad", "eou", "forced"])
    sp_ab.add_argument("--llm-delay", type=float, default=0.0)
    sp_ab.add_argument("--voice", default=None)
    sp_ab.add_argument("--chunk-ms", type=int, default=250)
    sp_ab.add_argument("--vad-backend", choices=["silero", "webrtcvad"], default="silero",
                       help="server-side VAD query param for --eos vad on seeed-local-voice")
    sp_ab.add_argument("--vad-silence-ms", type=int, default=400)
    sp_ab.add_argument("--realtime", action="store_true", default=True)
    sp_ab.add_argument("--no-realtime", dest="realtime", action="store_false")
    sp_ab.set_defaults(func=cmd_ab)

    sp_noise = sub.add_parser("noise"); add_common(sp_noise)
    sp_noise.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5, 0],
                          help="SNR levels in dB (lower = noisier)")
    sp_noise.add_argument("--noise-type", choices=["white", "pink", "babble"],
                          default="babble")
    sp_noise.set_defaults(func=cmd_noise)

    sp_stab = sub.add_parser("stability")
    sp_stab.add_argument("--base-url", default="http://localhost:8000")
    sp_stab.add_argument("--container", default=None)
    sp_stab.add_argument("--duration-min", type=float, default=30.0)
    sp_stab.add_argument("--mode", choices=["asr", "tts", "v2v"], default="v2v")
    sp_stab.set_defaults(func=cmd_stability)

    sp_clone = sub.add_parser("clone")
    sp_clone.add_argument("--base-url", default="http://localhost:8000")
    sp_clone.add_argument("--container", default=None)
    sp_clone.add_argument("--refs", required=True,
                          help="dir of reference voice WAVs (e.g., bench/perf/corpus/voices)")
    sp_clone.add_argument("--texts", default=str(
        Path(__file__).resolve().parent / "corpus" / "clone_texts.json"))
    sp_clone.add_argument("--warmup", type=int, default=1)
    sp_clone.add_argument("--runs", type=int, default=5)
    sp_clone.add_argument("--skip-similarity", action="store_true",
                          help="skip resemblyzer-based speaker similarity (faster)")
    sp_clone.set_defaults(func=cmd_clone)

    sp_boot = sub.add_parser("boot")
    sp_boot.add_argument("--base-url", default="http://localhost:8000")
    sp_boot.add_argument("--container", required=True)
    sp_boot.add_argument("--runs", type=int, default=3)
    sp_boot.add_argument("--health-path", default="/health")
    sp_boot.add_argument("--timeout-s", type=float, default=300.0)
    sp_boot.set_defaults(func=cmd_boot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
