# services/telegram_bot.py
"""Telegram bot — phone-friendly entry point into the existing chat pipeline.

Flow:
    user sends a text message
        → bot looks up (or creates) a persistent chat session for that chat_id
        → reuses build_chat_context() + llm_call_async() to get a reply
        → converts the reply to speech via the existing TTS service
        → sends a native voice note (OGG/Opus) back to the user

Reuses:
    - routes.chat_helpers.build_chat_context   (full preface / memory / RAG / harness)
    - src.llm_core.llm_call_async              (non-streaming chat call)
    - services.tts.tts_service.get_tts_service (synthesize → WAV)
    - session_manager                          (one persistent session per chat)

Long-polling (no public URL or webhook needed). Activated only when
TELEGRAM_BOT_TOKEN is set in .env, so the rest of the app is unaffected when
the feature is disabled.
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Persistent mapping: telegram chat_id → odysseus session_id. One persistent
# session per Telegram chat keeps the conversation continuous (memory works,
# harness is applied, etc.) without spawning a new session for every message.
_SESSION_MAP_PATH = Path("data/telegram_sessions.json")


def _load_session_map() -> dict:
    try:
        if _SESSION_MAP_PATH.is_file():
            return json.loads(_SESSION_MAP_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read telegram session map: %s", e)
    return {}


def _save_session_map(mapping: dict) -> None:
    try:
        _SESSION_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — don't leave a half-written file if the process dies.
        tmp = _SESSION_MAP_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        tmp.replace(_SESSION_MAP_PATH)
    except Exception as e:
        logger.warning("Failed to persist telegram session map: %s", e)


# ── Configuration ──────────────────────────────────────────────────────── #

@dataclass
class TelegramConfig:
    bot_token: str
    allowed_chat_id: str  # "" = allow any chat
    owner: str            # odysseus username to attribute messages to


def load_telegram_config() -> Optional[TelegramConfig]:
    """Read TELEGRAM_BOT_TOKEN / TELEGRAM_ALLOWED_CHAT_ID / TELEGRAM_OWNER
    from .env. Returns None if no token is set (bot is disabled)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    allowed = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID", "").strip()
    # Default to single-user mode (no auth) so the bot works out of the box
    # in dev setups. Once the user turns auth on, set TELEGRAM_OWNER to the
    # username that should own the chat history.
    owner = os.environ.get("TELEGRAM_OWNER", "").strip() or ""
    return TelegramConfig(bot_token=token, allowed_chat_id=allowed, owner=owner)


# ── Helpers ────────────────────────────────────────────────────────────── #

def _wav_to_ogg_opus(wav_bytes: bytes) -> Optional[bytes]:
    """Convert WAV bytes → OGG/Opus via ffmpeg. Telegram's sendVoice() only
    accepts OGG/Opus (or a couple of other formats) — WAV plays but is
    delivered as a generic file, not a voice-note bubble. Returns None on
    ffmpeg failure so the caller can fall back to a text reply."""
    if not wav_bytes:
        return None
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.warning("ffmpeg not found on PATH; cannot send voice note")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(wav_bytes)
            wav_path = wf.name
        ogg_path = wav_path.replace(".wav", ".ogg")
        # -c:a libopus → OGG/Opus, the format Telegram wants for voice notes.
        # -b:a 32k → bitrate tuned for speech (mono voice).
        # -ac 1 → mono. -ar 24000 matches Piper's native rate for amy-medium;
        # if a different voice is selected Piper will resample on the way in.
        # -vbr on + -application voip → opinionated about being a voice stream.
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", wav_path,
            "-c:a", "libopus",
            "-b:a", "32k",
            "-ac", "1",
            "-application", "voip",
            "-vbr", "on",
            ogg_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            logger.error("ffmpeg failed: %s", proc.stderr.strip()[:300])
            return None
        with open(ogg_path, "rb") as f:
            ogg_bytes = f.read()
        return ogg_bytes
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out")
        return None
    except Exception as e:
        logger.error("WAV→OGG conversion failed: %s", e)
        return None
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass
        try:
            os.unlink(ogg_path)
        except Exception:
            pass


def _find_ffmpeg() -> Optional[str]:
    """Find ffmpeg on PATH. We don't ship it as a dep (it's a system tool),
    but it's universally available on the platforms this app targets."""
    from shutil import which
    return which("ffmpeg")


# ── Bot ────────────────────────────────────────────────────────────────── #

class TelegramBot:
    """Long-polling Telegram bot. Lazily constructs the Application on start()
    so importing this module is cheap and the app can boot without Telegram
    deps installed."""

    def __init__(self, config: TelegramConfig, app_deps: dict):
        """
        config: tokens + allowed chat ids
        app_deps: dict with keys — session_manager, chat_handler, chat_processor,
            memory_manager, memory_vector, research_handler, upload_handler,
            preset_manager, skills_manager, webhook_manager, auth_manager
        """
        self.config = config
        self.deps = app_deps
        self._app = None          # PTB Application
        self._task: Optional[asyncio.Task] = None
        self._ready = False
        self._start_lock = asyncio.Lock()

    # ── Lifecycle ──

    async def start(self) -> bool:
        """Start the long-polling loop. Returns True on success, False if
        startup failed (so the app logs the warning instead of crashing)."""
        async with self._start_lock:
            if self._task and not self._task.done():
                return True
            try:
                from telegram.ext import (
                    ApplicationBuilder, MessageHandler, CommandHandler, filters,
                )
            except ImportError as e:
                logger.error(
                    "python-telegram-bot is not installed. "
                    "Run: pip install python-telegram-bot — bot disabled."
                )
                logger.error("ImportError: %s", e)
                return False

            try:
                builder = ApplicationBuilder().token(self.config.bot_token)
                # Reduce noisy heartbeat logs — Info-level only on first start.
                builder = builder.connect_timeout(20).read_timeout(20)
                self._app = builder.build()

                # /start — friendly onboarding, no LLM call.
                self._app.add_handler(
                    CommandHandler("start", self._cmd_start)
                )
                # /reset — start a fresh session for this chat.
                self._app.add_handler(
                    CommandHandler("reset", self._cmd_reset)
                )
                # /voice off | on — disable/enable voice replies per-chat.
                self._app.add_handler(
                    CommandHandler("voice", self._cmd_voice)
                )
                # Text messages go through the chat pipeline.
                self._app.add_handler(
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
                )

                await self._app.initialize()
                await self._app.start()
                # Updater is started by start_polling(); we offload it to a
                # background task so it doesn't block app startup.
                self._task = asyncio.create_task(self._app.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=["message"],
                ))
                self._ready = True
                logger.info(
                    "Telegram bot started (allowed_chat_id=%s, owner=%s)",
                    self.config.allowed_chat_id or "ANY", self.config.owner or "<unauth>",
                )
                return True
            except Exception as e:
                logger.error("Telegram bot startup failed: %s", e, exc_info=True)
                return False

    async def stop(self) -> None:
        """Stop the long-polling loop. Safe to call multiple times."""
        if not self._app:
            return
        try:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
        except Exception as e:
            logger.debug("Telegram updater stop: %s", e)
        try:
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            logger.debug("Telegram app stop: %s", e)
        self._app = None
        self._ready = False
        self._task = None

    @property
    def is_running(self) -> bool:
        return self._ready and self._task is not None and not self._task.done()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "allowed_chat_id": self.config.allowed_chat_id or None,
            "owner": self.config.owner or None,
            "voice_reply": self._voice_reply_enabled(),
        }

    # ── Per-chat state ──

    def _voice_reply_enabled(self) -> bool:
        # Voice reply is on by default; user can toggle with /voice off
        # at runtime. Persisted in the session map alongside session_id.
        return True  # default; per-chat state is stored separately

    def _get_chat_state(self, chat_id: int) -> dict:
        """Per-chat runtime state: voice-on/off, current session_id."""
        # Reuse the session map file for both: session_id + voice flag.
        # The session_id is the value; voice flag is a per-key sub-object.
        # For now keep it simple: store voice flag in a sidecar file.
        path = Path("data/telegram_chat_state.json")
        try:
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8"))
                return state.get(str(chat_id), {"voice": True})
        except Exception:
            pass
        return {"voice": True}

    def _set_chat_state(self, chat_id: int, **fields) -> None:
        path = Path("data/telegram_chat_state.json")
        try:
            state = {}
            if path.is_file():
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}
            cur = dict(state.get(str(chat_id), {"voice": True}))
            cur.update(fields)
            state[str(chat_id)] = cur
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            logger.warning("Failed to persist telegram chat state: %s", e)

    # ── Authorization ──

    def _is_authorized(self, chat_id: int) -> bool:
        if not self.config.allowed_chat_id:
            return True  # no allowlist configured
        try:
            return str(chat_id) == str(self.config.allowed_chat_id)
        except Exception:
            return False

    # ── Session management ──

    def _get_or_create_session(self, chat_id: int) -> Optional[Any]:
        """Look up the existing session for this chat, or create one using
        the user's default endpoint/model. Returns None if no default is
        configured (the bot tells the user to set one up)."""
        sm = self.deps.get("session_manager")
        if sm is None:
            return None

        mapping = _load_session_map()
        chat_key = str(chat_id)
        existing_id = mapping.get(chat_key)
        if existing_id:
            try:
                return sm.get_session(existing_id)
            except Exception:
                # Stale id in the map — drop it and fall through to create.
                mapping.pop(chat_key, None)
                _save_session_map(mapping)

        # Create a new session using the configured default endpoint/model.
        ep_id, ep_url, ep_model, ep_headers = self._resolve_default_endpoint()
        if not ep_url or not ep_model:
            return None
        new_id = f"tg-{chat_id}-{uuid.uuid4().hex[:8]}"
        try:
            sess = sm.create_session(
                session_id=new_id,
                name=f"Telegram: {chat_id}",
                endpoint_url=ep_url,
                model=ep_model,
                rag=False,
                owner=self.config.owner or None,
            )
            sess.headers = ep_headers or {}
        except Exception as e:
            logger.error("Failed to create telegram session: %s", e)
            return None

        mapping[chat_key] = new_id
        _save_session_map(mapping)
        return sess

    def _resolve_default_endpoint(self) -> tuple:
        """Resolve the default chat endpoint to (id, url, model, headers)."""
        try:
            from src.endpoint_resolver import resolve_endpoint
            url, model, headers = resolve_endpoint(
                setting_prefix="default",
                fallback_url="",
                fallback_model="",
                fallback_headers={},
                owner=self.config.owner or None,
            )
            # The 4th value (endpoint_id) isn't returned by resolve_endpoint
            # — that's fine, we just need url+model+headers here.
            return "", url, model, headers or {}
        except Exception as e:
            logger.warning("resolve_endpoint failed: %s", e)
            return "", "", "", {}

    def _forget_session(self, chat_id: int) -> None:
        mapping = _load_session_map()
        chat_key = str(chat_id)
        sess_id = mapping.pop(chat_key, None)
        if sess_id:
            _save_session_map(mapping)
        # Also drop the DB row so the session list doesn't accumulate dead
        # telegram sessions forever.
        if sess_id and self.deps.get("session_manager"):
            try:
                sm = self.deps["session_manager"]
                if hasattr(sm, "delete_session"):
                    sm.delete_session(sess_id)
            except Exception as e:
                logger.debug("Failed to delete old telegram session: %s", e)

    # ── Chat dispatch ──

    async def _on_text(self, update, context):
        """Main handler: text in → context build → LLM call → TTS → voice out."""
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message or not message.text:
            return
        chat_id = chat.id
        if not self._is_authorized(chat_id):
            logger.warning("Unauthorized telegram chat_id=%s; ignoring", chat_id)
            return

        text = (message.text or "").strip()
        if not text:
            return

        # Typing indicator — gives the user a heartbeat that the bot is alive
        # while the LLM + TTS run (a few seconds for short replies).
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass

        # Resolve session
        sess = self._get_or_create_session(chat_id)
        if not sess:
            await message.reply_text(
                "No default model is configured yet. Open the Odysseus web UI "
                "→ Settings → Default Chat Model, pick a model, then send a "
                "message here again."
            )
            return

        session_id = sess.id
        deps = self.deps
        chat_handler = deps.get("chat_handler")
        chat_processor = deps.get("chat_processor")
        if not chat_handler or not chat_processor:
            await message.reply_text("Bot misconfigured (chat pipeline missing).")
            return

        # Build a minimal stand-in for FastAPI's Request so build_chat_context
        # can call get_current_user(request). We only need state.current_user.
        class _Req:
            def __init__(self, owner):
                self.state = type("_S", (), {})()
                self.state.current_user = owner or None
                self.app = type("_A", (), {})()
                self.app.state = type("_AS", (), {})()
                # Preserve app.state.auth_manager for privilege checks
                _am = getattr(deps.get("auth_manager"), "__class__", None)
                if deps.get("auth_manager"):
                    self.app.state.auth_manager = deps["auth_manager"]
        request = _Req(self.config.owner or None)

        try:
            from routes.chat_helpers import build_chat_context, run_post_response_tasks
            from core.models import ChatMessage
            ctx = await build_chat_context(
                sess, request, chat_handler, chat_processor,
                message=text,
                session_id=session_id,
                # Use the most basic chat defaults for telegram; the user can
                # always override per-user in the web UI's Settings panel.
                incognito=False,
                no_memory=False,
                use_web=False,
                use_rag=True,
                use_enhanced_message=False,
            )
        except Exception as e:
            logger.error("build_chat_context failed for telegram: %s", e, exc_info=True)
            await message.reply_text(
                f"Sorry, I couldn't process that ({type(e).__name__})."
            )
            return

        # Call the LLM
        try:
            from src.llm_core import llm_call_async
            reply = await llm_call_async(
                sess.endpoint_url,
                sess.model,
                ctx.messages,
                headers=sess.headers,
                temperature=ctx.preset.temperature,
                max_tokens=ctx.preset.max_tokens,
            )
        except Exception as e:
            logger.error("LLM call failed for telegram: %s", e, exc_info=True)
            await message.reply_text(
                f"Sorry, the model didn't respond ({type(e).__name__})."
            )
            return

        # Save the reply into session history (so context accumulates).
        try:
            from routes.chat_helpers import save_assistant_response
            from routes.chat_helpers import run_post_response_tasks
            save_assistant_response(
                sess, deps.get("session_manager"), session_id, reply, None,
                character_name=ctx.preset.character_name,
                web_sources=ctx.web_sources,
                rag_sources=ctx.rag_sources,
                used_memories=ctx.used_memories,
            )
            # Fire post-response tasks (auto-name, webhooks). The memory
            # extractor and webhook are best-effort — wrap to keep the bot
            # alive on any background error.
            try:
                run_post_response_tasks(
                    sess, deps.get("session_manager"), session_id,
                    text, reply, None,
                    ctx.uprefs,
                    deps.get("memory_manager"),
                    deps.get("memory_vector"),
                    deps.get("webhook_manager"),
                    character_name=ctx.preset.character_name,
                    owner=ctx.user,
                )
            except Exception as _e:
                logger.debug("run_post_response_tasks (telegram): %s", _e)
        except Exception as e:
            logger.warning("Failed to persist telegram reply: %s", e)

        # Trim display text — Telegram caps messages at 4096 chars.
        display_text = (reply or "").strip()
        if len(display_text) > 4000:
            display_text = display_text[:3997] + "..."

        # Voice reply if enabled AND the response is non-trivial. Very short
        # replies ("OK", "Done.") are sent as text only — sending a 1-second
        # voice note for a single-word answer is annoying.
        voice_on = self._get_chat_state(chat_id).get("voice", True)
        sent_voice = False
        if voice_on and len(display_text) >= 12:
            try:
                sent_voice = await self._send_voice_reply(chat_id, display_text, context)
            except Exception as e:
                logger.error("Voice reply failed: %s", e)
        # Always send text too, so the user has a readable copy. Telegram
        # shows voice + caption in the same bubble when both are sent, but
        # sendVoice + sendMessage are two separate API calls and the cleaner
        # UX is to send the voice note with a caption (or send text first,
        # then voice — which is what we do here so the user sees the text
        # immediately while the audio renders).
        try:
            await message.reply_text(display_text or "...")
        except Exception as e:
            logger.warning("Text reply failed: %s", e)

    async def _send_voice_reply(self, chat_id: int, text: str, context) -> bool:
        """Synthesize text via TTS, convert WAV→OGG/Opus, and send as a
        native voice note (sendVoice). Returns True if a voice note was
        sent, False if we fell back (caller should have already sent text)."""
        from services.tts.tts_service import get_tts_service

        tts = get_tts_service()
        if not tts.available:
            return False

        # Run synthesis in a thread so we don't block the event loop on
        # the first request (Piper ONNX load can take ~1-2s).
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, tts.synthesize, text)
        if not wav_bytes:
            return False

        ogg_bytes = await loop.run_in_executor(None, _wav_to_ogg_opus, wav_bytes)
        if not ogg_bytes:
            return False

        # send_voice requires a file-like object or path. Use BytesIO so we
        # don't litter the disk with one-off files.
        try:
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=io.BytesIO(ogg_bytes),
                # Disable notification — voice notes are conversational,
                # not attention-grabbing alerts.
                disable_notification=True,
            )
            return True
        except Exception as e:
            logger.error("sendVoice failed: %s", e)
            return False

    # ── Commands ──

    async def _cmd_start(self, update, context):
        chat = update.effective_chat
        if not chat or not self._is_authorized(chat.id):
            return
        await update.effective_message.reply_text(
            "Hey — I'm Rose, on Odysseus. Send me a message and I'll reply "
            "with a voice note. /reset to start a new conversation, "
            "/voice off to switch to text-only."
        )

    async def _cmd_reset(self, update, context):
        chat = update.effective_chat
        if not chat or not self._is_authorized(chat.id):
            return
        self._forget_session(chat.id)
        await update.effective_message.reply_text(
            "Conversation reset. Your next message starts a new chat."
        )

    async def _cmd_voice(self, update, context):
        chat = update.effective_chat
        if not chat or not self._is_authorized(chat.id):
            return
        cur = self._get_chat_state(chat.id).get("voice", True)
        # Allow `/voice off`, `/voice on`, or just `/voice` (toggle).
        args = (context.args or []) if context else []
        if args:
            arg = args[0].lower().strip()
            if arg in ("on", "1", "true", "yes", "enable"):
                new_val = True
            elif arg in ("off", "0", "false", "no", "disable"):
                new_val = False
            else:
                await update.effective_message.reply_text(
                    "Usage: /voice on, /voice off, or just /voice to toggle."
                )
                return
        else:
            new_val = not cur
        self._set_chat_state(chat.id, voice=new_val)
        await update.effective_message.reply_text(
            f"Voice replies are now {'on' if new_val else 'off'}."
        )


# ── Module-level singleton ─────────────────────────────────────────────── #

_bot: Optional[TelegramBot] = None


def init_telegram_bot(app_deps: dict) -> Optional[TelegramBot]:
    """Initialize the bot from current env. Returns None if no token is set."""
    global _bot
    if _bot is not None:
        return _bot
    cfg = load_telegram_config()
    if cfg is None:
        return None
    _bot = TelegramBot(cfg, app_deps)
    return _bot


def get_telegram_bot() -> Optional[TelegramBot]:
    return _bot
