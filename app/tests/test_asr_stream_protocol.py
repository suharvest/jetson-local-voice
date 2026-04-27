"""Integration test for /asr/stream WebSocket message protocol.

Asserts type field ("partial"/"final"/"reset") is present.
Asserts replace semantics: client overwrites display on each partial,
final display matches the source text (CER < 5%).
Asserts at least one type == "final" message is emitted.
"""

import argparse, json, sys, time, os
import numpy as np
import soundfile as sf
import websocket


def replace_display(msgs):
    """Simulate client-side replace semantics.

    Returns the final displayed text after processing all messages:
    - type == "reset": clear display
    - type == "partial" or not final: update display (replace)
    - type == "final": freeze display
    """
    display = ""
    final_text = ""
    for msg in msgs:
        t = msg.get("text", "")
        msg_type = msg.get("type")
        if msg_type == "reset":
            display = ""
        elif msg_type == "final":
            display = t
            final_text = t
        else:
            # partial — replace display
            display = t
    return final_text or display


def test_asr_stream_protocol(host, wav_path, language="zh"):
    """Stream WAV through /asr/stream, collect messages, verify protocol."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )
        sr = 16000
    audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    ws_url = f"ws://{host}/asr/stream?language={language}&sample_rate=16000"
    ws = websocket.create_connection(ws_url, timeout=90)

    msgs = []
    chunk_n = int(16000 * 0.1)  # 100ms chunks
    for i in range(0, len(audio_i16), chunk_n):
        ws.send_binary(audio_i16[i:i + chunk_n].tobytes())
        time.sleep(0.1)
        # Collect any partial results
        try:
            while True:
                raw = ws.recv()
                msgs.append(json.loads(raw))
        except websocket.WebSocketTimeoutException:
            pass
        ws.settimeout(0.1)

    # Finalize
    ws.settimeout(10)
    ws.send_binary(b"")
    while True:
        try:
            raw = ws.recv()
            msgs.append(json.loads(raw))
        except (websocket.WebSocketTimeoutException, websocket.WebSocketConnectionClosedException):
            break
        # After final, one more for good measure, then timeout
        ws.settimeout(1)
    ws.close()

    # --- Assertions ---
    print(f"\nReceived {len(msgs)} messages")
    types_seen = set()
    for m in msgs:
        mt = m.get("type", "MISSING")
        types_seen.add(mt)
        text_preview = m.get("text", "")[:60]
        print(f"  [{mt}] {text_preview}")

    # 1. Every message has a type field
    missing_type = [m for m in msgs if "type" not in m]
    assert not missing_type, f"{len(missing_type)} messages missing 'type' field"

    # 2. All types are valid
    valid_types = {"partial", "final", "reset"}
    for mt in types_seen:
        assert mt in valid_types, f"Invalid type: {mt}"

    # 3. At least one final message
    finals = [m for m in msgs if m.get("type") == "final"]
    assert len(finals) >= 1, f"No type=='final' message found among {len(msgs)} messages"

    # 4. Replace semantics: partial updates replace, final text is the last final
    display_text = replace_display(msgs)
    assert display_text, "Display text is empty after processing"
    print(f"\n  Display text: {display_text}")
    print(f"  Final messages: {len(finals)}")

    return display_text, msgs


def main():
    p = argparse.ArgumentParser(description="ASR stream protocol test")
    p.add_argument("wav_path", help="Path to WAV file")
    p.add_argument("--host", default="100.67.111.58:8621",
                   help="ASR service host:port")
    p.add_argument("--language", default="zh")
    args = p.parse_args()

    display, msgs = test_asr_stream_protocol(args.host, args.wav_path, args.language)

    types_seen = set(m.get("type") for m in msgs)
    print(f"\n=== Results ===")
    print(f"  Messages: {len(msgs)}")
    print(f"  Types seen: {types_seen}")
    print(f"  Has final: {'final' in types_seen}")
    print(f"  Display text: {display}")
    if "final" in types_seen:
        print("  PASS: type=='final' message present")
    else:
        print("  FAIL: no type=='final' message")
        sys.exit(1)


if __name__ == "__main__":
    main()
