"""Headless Chromium check of the dashboard rendering after one turn."""
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_dashboard_renders_after_turn(test_config):
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright not installed; run `uv sync --extra e2e`")

    audio = ScriptedAudioIO([(800, WAV_DIR / "hello.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        # Drive one full turn.
        await probe.wait_event("on_user_utterance", timeout=20)
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=30)

        port = test_config.metadata["dashboard_port"]
        try:
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.launch(headless=True)
                except Exception as e:
                    pytest.skip(
                        f"Chromium not installed; run `uv run python -m playwright install chromium`: {e}"
                    )
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
                # Trigger a refresh-from-snapshot by opening the WS the page does.
                await page.wait_for_selector("#statePill", timeout=8000)
                pill = (await page.text_content("#statePill")) or ""
                assert any(tok in pill for tok in ("待机", "IDLE", "idle", "等待")), (
                    f"unexpected state pill text: {pill!r}"
                )
                # Chat tab should at minimum exist.
                chat_html = await page.content()
                assert "statePill" in chat_html
                await page.screenshot(path=f"/tmp/dashboard_{port}.png")
                await browser.close()
        except Exception:
            raise
