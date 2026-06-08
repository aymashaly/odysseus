# src/tts_service.py
"""Multi-provider TTS service — dispatches to local Kokoro, OpenAI-compatible API, or browser."""

import io
import os
import wave
import logging
import hashlib
import httpx
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _safe_speed(value, default: float = 1.0) -> float:
    """Parse the stored tts_speed defensively. The settings layer tolerates
    corrupt/agent-written config, so a non-numeric or empty value (e.g. an agent
    setting "speech speed" = "fast", or a hand-edited settings.json) must not
    crash synthesis or the stats endpoint with a ValueError."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return default
    return speed if speed > 0 else default


class TTSService:
    """Multi-provider TTS service.

    Reads provider config from data/settings.json on each call.
    Providers:
      "disabled"        — no TTS
      "browser"         — client-side Web Speech API (no server synthesis)
      "local"           — Kokoro-82M on GPU
      "endpoint:<id>"   — OpenAI-compatible /audio/speech via ModelEndpoint
    """

    def __init__(self, cache_dir: str = "data/tts_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._kokoro = None  # lazy-init
        self._piper = None   # lazy-init

    # ── Settings ──

    def _load_settings(self) -> dict:
        from src.settings import load_settings
        saved = load_settings()
        return {
            "tts_enabled": saved.get("tts_enabled", True),
            "tts_provider": saved.get("tts_provider", "disabled"),
            "tts_model": saved.get("tts_model", "tts-1"),
            "tts_voice": saved.get("tts_voice", "alloy"),
            "tts_speed": saved.get("tts_speed", "1"),
            "tts_piper_voice": saved.get("tts_piper_voice", "en_US-amy-medium"),
            "tts_piper_length_scale": saved.get("tts_piper_length_scale", 1.0),
            "tts_piper_noise_scale": saved.get("tts_piper_noise_scale", 0.667),
            "tts_piper_noise_w": saved.get("tts_piper_noise_w", 0.8),
        }

    @property
    def available(self) -> bool:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return False
        provider = settings["tts_provider"]
        if provider == "disabled":
            return False
        if provider == "browser":
            return True  # handled client-side
        if provider == "local":
            kokoro = self._get_kokoro()
            return kokoro is not None and kokoro.available
        if provider == "piper":
            piper = self._get_piper()
            return piper is not None and piper.available
        if provider.startswith("endpoint:"):
            return True  # assume reachable; errors surface at synthesis time
        return False

    # ── Cache ──

    def _cache_key(self, text: str, provider: str, model: str, voice: str, speed: float = 1.0) -> str:
        raw = f"{provider}|{model}|{voice}|{speed}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[bytes]:
        for ext in (".mp3", ".wav"):
            path = self.cache_dir / f"{key}{ext}"
            if path.exists():
                return path.read_bytes()
        return None

    def _put_cache(self, key: str, data: bytes):
        ext = ".mp3" if (len(data) >= 3 and (data[:3] == b'ID3' or (data[0] == 0xff and (data[1] & 0xe0) == 0xe0))) else ".wav"
        (self.cache_dir / f"{key}{ext}").write_bytes(data)

    def clear_cache(self):
        count = 0
        for f in self.cache_dir.glob("*.*"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached TTS files")

    # ── Kokoro (local) ──

    def _get_kokoro(self):
        if self._kokoro is None:
            self._kokoro = _KokoroPipeline()
        return self._kokoro

    # ── Piper (local CPU) ──

    def _get_piper(self):
        if self._piper is None:
            self._piper = _PiperPipeline()
        return self._piper

    # ── API endpoint ──

    def _synthesize_api(self, text: str, endpoint_id: str, model: str, voice: str, speed: float = 1.0) -> Optional[bytes]:
        from src.database import SessionLocal, ModelEndpoint

        db = SessionLocal()
        try:
            ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).first()
            if not ep:
                logger.error(f"TTS endpoint {endpoint_id} not found")
                return None
            base_url = ep.base_url.rstrip("/")
            api_key = ep.api_key
        finally:
            db.close()

        url = base_url + "/audio/speech"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed,
        }

        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            logger.info(f"API TTS: {len(r.content)} bytes from {base_url}")
            return r.content
        except Exception as e:
            logger.error(f"API TTS synthesis failed: {e}")
            return None

    # ── Public interface ──

    def synthesize(self, text: str, use_cache: bool = True) -> Optional[bytes]:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return None
        provider = settings["tts_provider"]
        model = settings["tts_model"]
        voice = settings["tts_voice"]
        speed = _safe_speed(settings.get("tts_speed", "1"))

        if provider in ("disabled", "browser"):
            return None

        if len(text) > 5000:
            text = text[:5000]

        if use_cache:
            key = self._cache_key(text, provider, model, voice, speed)
            cached = self._get_cached(key)
            if cached:
                logger.info(f"TTS cache hit ({len(text)} chars)")
                return cached

        audio_data = None

        if provider == "local":
            kokoro = self._get_kokoro()
            if kokoro and kokoro.available:
                audio_data = kokoro.synthesize_raw(text, voice)
            else:
                logger.warning("Kokoro TTS not available")
                return None
        elif provider == "piper":
            piper = self._get_piper()
            if piper and piper.available:
                settings_full = self._load_settings()
                audio_data = piper.synthesize_raw(
                    text,
                    voice=settings_full.get("tts_piper_voice", "en_US-amy-medium"),
                    length_scale=settings_full.get("tts_piper_length_scale", 1.0),
                    noise_scale=settings_full.get("tts_piper_noise_scale", 0.667),
                    noise_w=settings_full.get("tts_piper_noise_w", 0.8),
                )
            else:
                logger.warning("Piper TTS not available")
                return None
        elif provider.startswith("endpoint:"):
            endpoint_id = provider.split(":", 1)[1]
            audio_data = self._synthesize_api(text, endpoint_id, model, voice, speed)
        else:
            logger.error(f"Unknown TTS provider: {provider}")
            return None

        if audio_data and use_cache:
            key = self._cache_key(text, provider, model, voice, speed)
            self._put_cache(key, audio_data)

        return audio_data

    def synthesize_to_base64(self, text: str) -> Optional[str]:
        import base64
        audio = self.synthesize(text)
        if audio:
            return base64.b64encode(audio).decode("utf-8")
        return None

    def set_voice(self, voice: str):
        """Legacy no-op — voice is now managed via admin settings."""

    def get_stats(self) -> Dict[str, Any]:
        settings = self._load_settings()
        provider = settings["tts_provider"]
        tts_enabled = settings.get("tts_enabled", True)

        cache_files = list(self.cache_dir.glob("*.wav")) + list(self.cache_dir.glob("*.mp3"))
        cache_size = sum(f.stat().st_size for f in cache_files)

        is_available = self.available and tts_enabled
        stats = {
            "available": is_available,
            "ready": is_available,
            "provider": provider,
            "model": settings["tts_model"],
            "voice": settings["tts_voice"],
            "speed": _safe_speed(settings.get("tts_speed", "1")),
            "cache_entries": len(cache_files),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        }

        if provider == "local":
            kokoro = self._get_kokoro()
            stats["model"] = "Kokoro-82M (GPU)" if (kokoro and kokoro.available) else "Kokoro (not loaded)"
        elif provider == "browser":
            stats["model"] = "Browser (Web Speech API)"
        elif provider == "piper":
            piper = self._get_piper()
            stats["model"] = (
                f"Piper {settings.get('tts_piper_voice', 'en_US-amy-medium')} (CPU)"
                if (piper and piper.available) else "Piper (not loaded)"
            )
        elif provider.startswith("endpoint:"):
            stats["endpoint_id"] = provider.split(":", 1)[1]

        return stats


class _KokoroPipeline:
    """Encapsulates the Kokoro-82M local GPU pipeline."""

    def __init__(self):
        self.pipeline = None
        self.available = False
        self.device = None
        self._init()

    def _init(self):
        try:
            import torch
            from kokoro import KPipeline

            if not torch.cuda.is_available():
                logger.warning("CUDA not available for Kokoro TTS")
                return

            self.device = torch.device("cuda:0")
            with torch.cuda.device(0):
                self.pipeline = KPipeline(lang_code="a")
                if hasattr(self.pipeline, "model"):
                    self.pipeline.model = self.pipeline.model.to(self.device)
            self.available = True
            logger.info("Kokoro-82M TTS pipeline loaded")
        except ImportError as e:
            logger.warning(f"Kokoro TTS not available: {e}")
            logger.warning("Install with: pip install kokoro soundfile")
        except Exception as e:
            logger.error(f"Kokoro init failed: {e}", exc_info=True)

    def synthesize_raw(self, text: str, voice: str = "af_heart") -> Optional[bytes]:
        if not self.available:
            return None
        try:
            import torch
            import numpy as np

            with torch.cuda.device(self.device):
                chunks = []
                for _, _, audio in self.pipeline(text, voice=voice):
                    chunks.append(audio)

            if not chunks:
                return None

            full = np.concatenate(chunks)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes((full * 32767).astype(np.int16).tobytes())
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            return None


class _PiperPipeline:
    """Encapsulates the Piper TTS local CPU pipeline.

    Piper is a fast, local, ONNX-based TTS system. Each voice is a small
    .onnx file (15-60 MB) plus a .onnx.json config. No GPU required, so it
    works on any machine. Used here as a CPU-friendly alternative to Kokoro.
    """

    # Default models are stored under data/piper_models/. Users can drop
    # additional .onnx / .onnx.json pairs there and reference them by stem.
    DEFAULT_VOICE = "en_US-amy-medium"
    MODELS_DIR = Path("data/piper_models")

    def __init__(self):
        self.voice = None
        self.config = None
        self.config_path = None
        self.model_path = None
        self.sample_rate = 22050
        self.available = False
        self._loaded_voice_name: Optional[str] = None
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        try:
            from piper import PiperVoice  # noqa: F401  (presence check)
        except ImportError as e:
            logger.warning("Piper TTS not available: %s", e)
            logger.warning("Install with: pip install piper-tts")
            return
        # We don't load the model here — load it lazily on first synthesize
        # so importing this module is cheap and only-imports-without-model
        # configs don't pay the model-load cost.
        self.available = True
        logger.info("Piper TTS runtime available (model will load on first use)")

    def _resolve_model(self, voice: str) -> Optional[tuple]:
        """Find the .onnx + .onnx.json pair for a voice name.

        Resolution order:
          1. Explicit path passed in (if `voice` contains a separator)
          2. <MODELS_DIR>/<voice>.onnx + .onnx.json
        """
        from piper import PiperVoice

        # If a path was passed, use it directly
        if os.sep in voice or "/" in voice:
            onnx = Path(voice)
            if onnx.is_file() and onnx.suffix == ".onnx":
                cfg = onnx.with_suffix(".onnx.json")
                if cfg.is_file():
                    return onnx, cfg
            logger.error("Piper voice path not found or missing config: %s", voice)
            return None

        # Default models dir
        onnx = self.MODELS_DIR / f"{voice}.onnx"
        cfg = self.MODELS_DIR / f"{voice}.onnx.json"
        if onnx.is_file() and cfg.is_file():
            return onnx, cfg

        logger.error(
            "Piper voice '%s' not found in %s. "
            "Download a voice from https://github.com/rhasspy/piper/releases "
            "and place the .onnx + .onnx.json files in that directory.",
            voice, self.MODELS_DIR,
        )
        return None

    def _ensure_loaded(self, voice: str) -> bool:
        """Load the model if not already loaded for this voice. Returns
        True if a usable model is in memory."""
        from piper import PiperVoice

        if self._loaded_voice_name == voice and self.voice is not None:
            return True

        paths = self._resolve_model(voice)
        if not paths:
            return False

        onnx_path, cfg_path = paths
        try:
            logger.info("Loading Piper voice '%s' from %s", voice, onnx_path)
            self.voice = PiperVoice.load(str(onnx_path), str(cfg_path))
            self.config_path = str(cfg_path)
            self.model_path = str(onnx_path)
            self.sample_rate = self.voice.config.sample_rate
            self._loaded_voice_name = voice
            logger.info("Piper voice '%s' loaded (sample_rate=%d)", voice, self.sample_rate)
            return True
        except Exception as e:
            logger.error("Failed to load Piper voice '%s': %s", voice, e, exc_info=True)
            self.voice = None
            self._loaded_voice_name = None
            return False

    def synthesize_raw(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> Optional[bytes]:
        if not self.available:
            return None
        if not text or not text.strip():
            return None
        if not self._ensure_loaded(voice):
            return None

        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w,
            )

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                # Piper streams chunks; we write each into the wav file as
                # it arrives. This keeps memory flat for long responses.
                for chunk in self.voice.synthesize(text, syn_config=syn_config):
                    wf.writeframes(chunk.audio_int16_bytes)
            return buf.getvalue()
        except Exception as e:
            logger.error("Piper synthesis failed: %s", e, exc_info=True)
            return None


# Module-level singleton
_tts_service = None

def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
