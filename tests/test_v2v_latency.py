"""V2V streaming latency test: EOS -> first TTS audio."""
import io, json, struct, time, wave, numpy as np, requests, websocket

def make_wav(text, session, base_url):
    resp = session.post(f"{base_url}/tts", json={"text": text}, timeout=120)
    resp.raise_for_status()
    return resp.content

def wav_to_pcm_chunks(wav_bytes, chunk_ms=100):
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    chunk_n = int(sr * chunk_ms / 1000)
    return [samples[i:i+chunk_n].tobytes() for i in range(0, len(samples), chunk_n)], sr

def wav_duration(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()

def run_v2v(session, base_url, wav_bytes, language="auto", realtime=True):
    chunks, sr = wav_to_pcm_chunks(wav_bytes)
    ws_url = base_url.replace("http://", "ws://")
    ws = websocket.create_connection(f"{ws_url}/asr/stream?language={language}&sample_rate={sr}", timeout=60)

    chunk_dur = 0.1  # 100ms chunks
    for c in chunks:
        ws.send_binary(c)
        if realtime:
            time.sleep(chunk_dur)

    t_eos = time.monotonic()
    ws.send_binary(b"")  # finalize signal
    final_text = ""
    while True:
        data = json.loads(ws.recv())
        if data.get("type") == "final":
            final_text = data.get("text", "").strip()
            break
        final_text = data.get("text", "").strip()
    t_asr = time.monotonic()
    ws.close()
    t_tts_req = time.monotonic()
    resp = session.post(f"{base_url}/tts/stream", json={"text": final_text}, stream=True, timeout=120)
    resp.raise_for_status()
    for chunk in resp.iter_content(chunk_size=4096):
        break
    t_tts_first = time.monotonic()
    for _ in resp.iter_content(chunk_size=8192):
        pass
    return {
        "audio_dur": wav_duration(wav_bytes), "text": final_text,
        "eos_to_audio_ms": (t_tts_first - t_eos) * 1000,
        "asr_ms": (t_asr - t_eos) * 1000,
        "tts_ttfb_ms": (t_tts_first - t_tts_req) * 1000,
        "handoff_ms": (t_tts_req - t_asr) * 1000,
    }

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost:8621")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--burst", action="store_true", help="Send all audio at once (not real-time)")
    args = parser.parse_args()
    base_url = f"http://{args.host}"
    session = requests.Session()
    health = session.get(f"{base_url}/health", timeout=5).json()
    print(f"ASR: {health.get('asr_backend')}  TTS: {health.get('tts_backend')}")
    mode = "burst" if args.burst else "realtime"
    print(f"Mode: {mode}")
    sentences = ["你好世界", "今天天气不错我们出去玩吧", "Hello world how are you today"]
    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {run+1}/{args.repeat} ---")
        print(f"\n{'Sentence':<30} {'Audio':>5} {'ASR':>7} {'Hand':>6} {'TTFB':>7} {'EOS>Aud':>8} Text")
        print("-" * 105)
        for sent in sentences:
            wav = make_wav(sent, session, base_url)
            lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sent) else "English"
            r = run_v2v(session, base_url, wav, language=lang, realtime=not args.burst)
            print(f"{sent:<30} {r['audio_dur']:>4.1f}s {r['asr_ms']:>6.0f}ms {r['handoff_ms']:>5.0f}ms {r['tts_ttfb_ms']:>6.0f}ms {r['eos_to_audio_ms']:>7.0f}ms {r['text']}")
    print("\nDONE")
