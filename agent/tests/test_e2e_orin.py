"""End-to-end agent layer validation, self-driven (no human mic)."""
import asyncio, io, json, os, struct, sys, time, wave
import numpy as np
import requests
from openai import AsyncOpenAI

sys.path.insert(0, "/Users/harvest/project/seeed-local-voice/agent/.venv/lib/python3.13/site-packages")
from websockets.asyncio.client import connect

os.environ["NO_PROXY"] = "100.82.225.102,localhost,127.0.0.1"
os.environ["no_proxy"] = "100.82.225.102,localhost,127.0.0.1"

ORIN = "100.82.225.102"
SLV_HTTP = f"http://{ORIN}:8621"
SLV_WS = f"ws://{ORIN}:8621/v2v/stream"
LLM_BASE = f"http://{ORIN}:8000/v1"
LLM_MODEL = "Qwen/Qwen3-4B-AWQ"

USER_TEXT = "你好，今天上海天气怎么样？"
SYSTEM = "你是一个简洁友善的语音助手。回答口语化，两三句话以内，不用 Markdown。"


def fetch_wav(text):
    from pathlib import Path
    wav_path = str(Path(__file__).parent / "fixtures" / "user_input.wav")
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    print(f"[setup] loaded {wav_path}: sr={sr}, samples={len(raw)//2}")
    return raw, sr


def resample_16k(pcm, sr):
    if sr == 16000:
        return pcm
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = int(len(s) * 16000 / sr)
    x0 = np.linspace(0, 1, len(s), endpoint=False)
    x1 = np.linspace(0, 1, n, endpoint=False)
    out = np.interp(x1, x0, s).astype(np.int16)
    print(f"[setup] resampled {sr}->16000Hz: {len(s)} -> {n} samples")
    return out.tobytes()


async def main():
    pcm, src_sr = fetch_wav(USER_TEXT)
    pcm = resample_16k(pcm, src_sr)
    chunk_bytes = 1600 * 2  # 100ms @ 16kHz int16
    chunks = [pcm[i:i+chunk_bytes] for i in range(0, len(pcm), chunk_bytes) if pcm[i:i+chunk_bytes]]
    print(f"[setup] {len(chunks)} chunks ({len(chunks)*0.1:.1f}s)")

    ws = await connect(SLV_WS, max_size=None)
    await ws.send(json.dumps({
        "type": "config",
        "asr_language": "zh", "tts_language": "zh",
        "sample_rate": 16000, "vad": "silero", "vad_silence_ms": 400,
        "multi_utterance": True,
    }))
    print("[1] WS+config sent")

    asr_final = None
    tts_sr = None
    tts_pcm = bytearray()
    tts_done = False

    async def reader():
        nonlocal asr_final, tts_sr, tts_done
        first_bin = True
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    if first_bin and len(msg) >= 4:
                        tts_sr = struct.unpack("<I", msg[:4])[0]
                        tts_pcm.extend(msg[4:])
                        print(f"[recv] FIRST tts bin: sr={tts_sr}, +{len(msg)-4}B")
                        first_bin = False
                    else:
                        tts_pcm.extend(msg)
                    continue
                try: j = json.loads(msg)
                except: continue
                t = j.get("type")
                if t == "asr_partial":
                    print(f"[recv] partial: {j.get('text','')[:60]!r}")
                elif t == "asr_final":
                    txt = j.get("text", "").strip()
                    print(f"[recv] FINAL: {txt!r}")
                    if txt and asr_final is None:
                        asr_final = txt
                elif t == "asr_endpoint": print(f"[recv] endpoint")
                elif t == "tts_started": print(f"[recv] tts_started: {j.get('sentence','')[:60]!r}")
                elif t == "tts_sentence_done": print(f"[recv] tts_sent_done: {j.get('sentence','')[:60]!r}")
                elif t == "tts_done":
                    tts_done = True
                    print(f"[recv] TTS_DONE")
                    break
                elif t == "error": print(f"[recv] ERROR: {j.get('error')}")
                else: print(f"[recv] {t}")
        except Exception as e:
            print(f"[recv] EXIT: {type(e).__name__}: {str(e)[:120]}")

    rt = asyncio.create_task(reader())

    t0 = time.time()
    for c in chunks:
        await ws.send(c)
        await asyncio.sleep(0.1)
    print(f"[2] pumped audio in {time.time()-t0:.1f}s")

    await ws.send(json.dumps({"type": "asr_eos"}))
    print(f"[3] asr_eos sent")

    tw = time.time()
    while asr_final is None and time.time() - tw < 8:
        await asyncio.sleep(0.1)
    if asr_final is None:
        print("[FAIL] no asr_final")
        rt.cancel(); await ws.close()
        return 1
    print(f"[4] got final in {time.time()-tw:.1f}s")

    llm = AsyncOpenAI(base_url=LLM_BASE, api_key="EMPTY")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": asr_final}]
    print(f"[5] LLM call: {asr_final!r}")
    t_llm = time.time()
    stream = await llm.chat.completions.create(
        model=LLM_MODEL, messages=messages, stream=True,
        extra_body={"save_system_prompt_kv_cache": True},
    )
    full = []
    n_tok = 0
    first_t = None
    async for chunk in stream:
        if not chunk.choices: continue
        d = chunk.choices[0].delta.content
        if not d: continue
        if first_t is None:
            first_t = time.time()
            print(f"[6] first token in {first_t-t_llm:.2f}s")
        full.append(d)
        await ws.send(json.dumps({"type": "text", "text": d}))
        n_tok += 1
    full_text = "".join(full)
    print(f"[7] LLM done: {n_tok} toks, '{full_text}'")

    await ws.send(json.dumps({"type": "tts_flush"}))
    print(f"[8] tts_flush sent")

    tw = time.time()
    while not tts_done and time.time() - tw < 30:
        await asyncio.sleep(0.1)
    print(f"[9] tts wait done in {time.time()-tw:.1f}s, got {len(tts_pcm)}B pcm")

    if tts_pcm and tts_sr:
        out = "/tmp/e2e_tts_output.wav"
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(tts_sr)
            wf.writeframes(bytes(tts_pcm))
        print(f"[10] saved {out}: {len(tts_pcm)//2/tts_sr:.2f}s @ {tts_sr}Hz")

    await ws.close()
    rt.cancel()
    await llm.close()

    ok = bool(asr_final and full_text and tts_pcm)
    print()
    print("=" * 60)
    print(f"INPUT  : {USER_TEXT!r}")
    print(f"ASR    : {asr_final!r}")
    print(f"LLM    : {full_text!r}")
    print(f"TTS OUT: {len(tts_pcm)//2/(tts_sr or 1):.2f}s of audio")
    print(f"RESULT : {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
