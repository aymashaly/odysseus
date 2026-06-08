# routes/telegram_routes.py
"""Admin endpoints for the Telegram bot — status, start, stop, send-test.

The bot itself is started/stopped by app.py's lifespan events. These routes
are an admin-only escape hatch: a quick way to verify wiring from the
admin panel or via curl, and to fire a test message without going through
Telegram.
"""

import asyncio
import logging
import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.auth_helpers import get_current_user
from services.telegram_bot import get_telegram_bot, load_telegram_config

logger = logging.getLogger(__name__)


def setup_telegram_routes() -> APIRouter:
    router = APIRouter(prefix="/api/telegram", tags=["telegram"])

    @router.get("/status")
    def status(request: Request):
        """Status snapshot — used by the admin panel to show whether the
        bot is connected and which chat is allowlisted."""
        _ = get_current_user(request)  # auth gate (no-op in single-user mode)
        bot = get_telegram_bot()
        cfg = load_telegram_config()
        # Whether the env even has a token. If not, the bot is permanently
        # disabled regardless of the runtime state.
        env_configured = bool(cfg and cfg.bot_token)
        if bot is None and env_configured:
            # Bot is in env but not yet started (startup race, or stopped).
            return {
                "running": False,
                "configured": True,
                "allowed_chat_id": cfg.allowed_chat_id or None,
                "owner": cfg.owner or None,
                "message": "Bot is configured but not running. Restart the app or POST /api/telegram/start.",
            }
        if bot is None:
            return {
                "running": False,
                "configured": False,
                "allowed_chat_id": None,
                "owner": None,
                "message": "TELEGRAM_BOT_TOKEN is not set — bot is disabled.",
            }
        st = bot.status()
        st["configured"] = True
        return st

    @router.post("/start")
    async def start_bot(request: Request):
        """Start the bot if it isn't already running. Idempotent — calling
        twice is a no-op."""
        _ = get_current_user(request)
        bot = get_telegram_bot()
        if bot is None:
            raise HTTPException(
                400,
                "TELEGRAM_BOT_TOKEN is not set in .env — bot cannot be started.",
            )
        ok = await bot.start()
        if not ok:
            raise HTTPException(500, "Bot failed to start — see server logs.")
        return {"ok": True, "running": True}

    @router.post("/stop")
    async def stop_bot(request: Request):
        """Stop the long-polling loop. Use this to silence the bot without
        restarting the app (e.g., for maintenance)."""
        _ = get_current_user(request)
        bot = get_telegram_bot()
        if bot is None:
            raise HTTPException(400, "Bot is not initialized.")
        await bot.stop()
        return {"ok": True, "running": False}

    @router.post("/test")
    async def test_tts(request: Request):
        """Synthesize a test phrase via the configured TTS provider and
        return it as base64-encoded WAV. Useful for verifying Piper
        without a Telegram chat. Bypasses the bot entirely — does NOT
        send a Telegram message."""
        _ = get_current_user(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        text = (body or {}).get("text", "").strip()
        if not text:
            text = "Hello from Odysseus."
        if len(text) > 1000:
            text = text[:1000]

        try:
            from services.tts.tts_service import get_tts_service
            tts = get_tts_service()
            stats = tts.get_stats()
            audio = tts.synthesize(text, use_cache=False)
            if not audio:
                raise HTTPException(
                    500,
                    f"TTS returned no audio. Provider={stats.get('provider')!r} "
                    f"available={stats.get('available')}. Check Piper voice files "
                    "in data/piper_models/ or set tts_provider to something else.",
                )
            import base64
            return {
                "ok": True,
                "provider": stats.get("provider"),
                "model": stats.get("model"),
                "voice": stats.get("voice"),
                "size_bytes": len(audio),
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "format": "wav",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Telegram test TTS failed: %s", e, exc_info=True)
            raise HTTPException(500, f"TTS failed: {e}")

    return router
