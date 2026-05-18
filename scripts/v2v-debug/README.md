# V2V Debug Dashboard

Standalone browser-based latency tester for the backend `/v2v/stream` endpoint.
Pure static HTML + a tiny stdlib static server. **Not shipped inside the backend
Docker image** — run it anywhere (laptop, dev box, or on the device itself) and
point it at any backend instance over WebSocket.

## What it does

- Captures your microphone, downsamples to 16 kHz int16 PCM, streams it to
  `/v2v/stream`
- Renders live `asr_partial` / `asr_final` events
- Optionally forwards the ASR final text through an OpenAI-compatible LLM
  endpoint (external like `https://api.openai.com/v1`, or a built-in compatible
  endpoint on the same host as the backend), then pipes the LLM tokens back
  into the same WS as TTS input
- Plays the streamed TTS PCM back through the browser
- Reports four latency tiles: first ASR partial / ASR final / LLM first token /
  first audio (all measured since session start)

Three modes:

| Mode | Pipeline |
|------|----------|
| 对话 | ASR → LLM chat reply → TTS |
| 同传 | ASR → LLM "translate to <ttsLang>" → TTS |
| 纯 ASR | ASR only (no TTS); if LLM enabled, shows translation/rewrite as text |

LLM source can be set to "无 LLM (回声)" to skip the LLM hop entirely — useful
for measuring pure ASR + TTS round-trip latency.

## Run the dashboard

```bash
uv run tools/v2v-debug/serve.py
# → http://localhost:8080  (auto-opens a browser tab)
```

Or any other static server:

```bash
python3 -m http.server 8080 --directory tools/v2v-debug
```

Options: `--port 8080`, `--host 0.0.0.0`, `--no-open`.

## Use it

1. Start the backend on some device. Default dev port is **9360**:

   ```bash
   docker compose up -d --build         # repo-root docker-compose.yml
   curl http://localhost:9360/health
   ```

2. Open the dashboard in a browser.
3. Edit the **WebSocket URL** field at the top of the left panel:
   - Local backend on the same machine: `ws://localhost:9360/v2v/stream`
   - Backend on a LAN device: `ws://192.168.x.x:9360/v2v/stream`
   - Production deploy (`deploy/docker-compose.yml`, port 8621):
     `ws://<host>:8621/v2v/stream`
4. Pick a mode, fill in the LLM section (or pick **无 LLM (回声)**), click
   **开始说话**, speak, click again to stop. Click **开始说话** again for the
   next round.

API keys you paste into the dashboard stay in the browser — the static server
never sees them, and they are sent directly to whatever LLM base URL you
configured.

## Files

- `index.html` — the whole UI + audio I/O + WS protocol logic, no build step
- `serve.py` — zero-dependency stdlib static server
