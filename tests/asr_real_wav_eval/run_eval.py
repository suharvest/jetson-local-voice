"""LibriSpeech ASR evaluation — stream real human FLACs through /asr/stream + WER.

Walks LibriSpeech dev-clean tree, picks N samples, reads FLAC + ground truth
from sibling .trans.txt, streams via WebSocket, computes WER.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import soundfile as sf
import websocket
import jiwer


def load_libri_samples(root: str, n: int = 15):
    """Return list of (flac_path, ground_truth_text). Walks speakers/chapters."""
    samples = []
    trans_files = sorted(glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True))
    for tf in trans_files:
        chap_dir = os.path.dirname(tf)
        trans = {}
        for line in open(tf):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                trans[parts[0]] = parts[1]
        # Take up to 2 from each chapter for diversity
        flacs = sorted(glob.glob(os.path.join(chap_dir, "*.flac")))[:2]
        for f in flacs:
            uid = os.path.splitext(os.path.basename(f))[0]
            if uid in trans:
                samples.append((f, trans[uid]))
            if len(samples) >= n:
                break
        if len(samples) >= n:
            break
    return samples[:n]


def resample_to_16k(audio, sr):
    if sr == 16000:
        return audio
    ratio = 16000 / sr
    new_len = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def send_via_ws(flac_path: str, host: str, language: str) -> str:
    audio, sr = sf.read(flac_path, dtype="float32")
    audio = resample_to_16k(audio, sr)
    audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    ws_url = f"ws://{host}/asr/stream?language={language}&sample_rate=16000"
    ws = websocket.create_connection(ws_url, timeout=90)
    chunk_ms = 100
    chunk_n = int(16000 * chunk_ms / 1000)
    for i in range(0, len(audio_i16), chunk_n):
        ws.send_binary(audio_i16[i:i + chunk_n].tobytes())
        time.sleep(chunk_ms / 1000)
    ws.send_binary(b"")
    final = ""
    while True:
        try:
            msg = ws.recv()
            data = json.loads(msg)
        except Exception:
            break
        if data.get("type") == "final":
            final = data.get("text", "").strip()
            break
    ws.close()
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="100.67.111.58:8621")
    p.add_argument("--libri-root", default=os.path.join(os.path.dirname(__file__),
                                                         "librispeech", "LibriSpeech", "dev-clean"))
    p.add_argument("--language", default="English")
    p.add_argument("-n", type=int, default=15)
    args = p.parse_args()

    samples = load_libri_samples(args.libri_root, args.n)
    if not samples:
        print(f"No samples found under {args.libri_root}")
        sys.exit(1)

    rows = []
    print(f"{'idx':<3} {'dur':>5} {'WER':>6}")
    print("-" * 120)
    for i, (flac, gt) in enumerate(samples):
        gt_norm = gt.lower().strip()
        info = sf.info(flac)
        dur = info.duration
        try:
            hyp = send_via_ws(flac, args.host, args.language).lower().strip()
        except Exception as e:
            hyp = f"<err: {e}>"
        wer = jiwer.wer(gt_norm, hyp) if not hyp.startswith("<err") else 1.0
        rows.append((i, dur, wer, gt_norm, hyp))
        print(f"{i:<3} {dur:>4.1f}s {wer:>5.0%}")
        print(f"    GT:  {gt_norm[:100]}")
        print(f"    HYP: {hyp[:100]}")
    print("-" * 120)
    wers = [r[2] for r in rows]
    print(f"\nn={len(wers)} median={np.median(wers):.0%} p90={np.quantile(wers, 0.9):.0%} "
          f"mean={np.mean(wers):.0%} max={max(wers):.0%}")
    print(f"WER above 10%: {sum(1 for w in wers if w > 0.1)}/{len(wers)}")
    print(f"WER above 25%: {sum(1 for w in wers if w > 0.25)}/{len(wers)}")


if __name__ == "__main__":
    main()
