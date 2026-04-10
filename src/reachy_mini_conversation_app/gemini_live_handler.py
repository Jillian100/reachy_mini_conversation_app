"""Gemini Live API handler for real-time audio/video conversations with Reachy Mini.

Drop-in replacement for OpenaiRealtimeHandler. Uses Google's Gemini Live API
for bidirectional audio streaming with native speech-to-speech capability.

Audio format: raw PCM, 16-bit little-endian, mono
Input (to Gemini):  16 kHz
Output (from Gemini): 24 kHz

Reference implementation: gamepop/reachy-mini-gemini (proven on Reachy Mini hardware)
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
import random
from typing import Any, Dict, Final, List, Optional, Tuple
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample

from google import genai
from google.genai import types

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# Debug: speech gate log to private file (was /tmp — moved 2026-04-10 per 007 audit).
# /tmp is world-readable and stores tool call payloads; new path is 600.
_gate_log_dir = os.path.expanduser("~/vicky_conversation/logs")
try:
    os.makedirs(_gate_log_dir, mode=0o700, exist_ok=True)
except Exception:
    pass
_gate_log_path = os.path.join(_gate_log_dir, "speech_gate.log")
_gate_fh = logging.FileHandler(_gate_log_path)
_gate_fh.setLevel(logging.DEBUG)
_gate_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_gate_fh)
logger.setLevel(logging.DEBUG)
try:
    os.chmod(_gate_log_path, 0o600)
except Exception:
    pass

# ── Vicky Voice Memory L1: crash-safe JSONL transcript hot log ──
# Minimal-invasion hook into metis_extensions/voice_memory/transcript_writer.py.
# Fails silent if module missing (e.g. on upstream merge without extensions).
try:
    import sys as _sys_tw
    _sys_tw.path.insert(
        0, os.path.expanduser("~/vicky_conversation/metis_extensions")
    )
    from voice_memory.transcript_writer import TranscriptWriter as _TW
    _transcript_writer = _TW.instance()
    logger.info(
        "Voice Memory L1: transcript writer ready at %s (session=%s)",
        _transcript_writer.log_dir, _transcript_writer.session_id,
    )
except Exception as _tw_err:
    _transcript_writer = None
    logger.warning("Voice Memory L1 disabled: %s", _tw_err)

# Gemini Live API audio parameters
GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000

# Default model — Gemini 3.1 Flash Live (launched 2026-03-26)
# Upgraded from 2.5 Flash native audio preview.
# Breaking changes vs 2.5:
#   - realtime_input.media_chunks → audio/video/text (separate Blob fields)
#   - send_client_content restricted to initial context only
#   - function calling is synchronous only
#   - thinkingBudget → thinkingLevel
#   - proactive audio & affective dialogue removed
DEFAULT_GEMINI_MODEL: Final[str] = "models/gemini-3.1-flash-live-preview"

# Volume gain for Gemini audio output (Gemini tends to output quieter audio)
VOLUME_GAIN: Final[float] = float(os.environ.get("GEMINI_VOLUME_GAIN", "3.0"))

# Energy gate: minimum RMS (float32 scale) to forward audio to Gemini.
# Filters ambient / distant speech so 圍棋 only responds to direct conversation.
# 0 = disabled (send everything). Typical direct speech ≈ 0.03–0.10, ambient ≈ 0.005–0.02.
ENERGY_GATE_THRESHOLD: Final[float] = float(os.environ.get("GEMINI_ENERGY_GATE", "0.06"))
# Hold time: keep forwarding audio for this many seconds after energy drops below threshold.
# Prevents cutting off pauses between words within a sentence.
ENERGY_GATE_HOLD_SEC: Final[float] = float(os.environ.get("GEMINI_ENERGY_HOLD", "2.0"))
# Pre-buffer: seconds of audio to keep in lookback ring buffer.
# When energy gate triggers, the pre-buffer is flushed first so speech onset isn't lost.
ENERGY_GATE_PREBUFFER_SEC: Final[float] = float(os.environ.get("GEMINI_ENERGY_PREBUFFER", "0.3"))
# Speech Band Energy Ratio (SBER): ratio of energy in 300-3400 Hz vs full spectrum.
# Human speech concentrates ~60-80% energy in this band; music spreads across full spectrum.
# Combined with RMS gate: audio must pass BOTH thresholds to reach Gemini.
SPEECH_BAND_RATIO_THRESHOLD: Final[float] = float(os.environ.get("GEMINI_SPEECH_BAND_RATIO", "0.5"))

# Conversation mode: two-tier dormant/active energy gate
ENERGY_GATE_DORMANT: Final[float] = float(os.environ.get("GEMINI_ENERGY_GATE_DORMANT", "0.12"))
# Motion effects: disable all auto-motions (tool-wait, micro-motion, thinking, listening nod)
# Gemini can still call play_emotion tool explicitly. OFF = quiet body.
# Motion effects level: 0=all off (guard), 1=minimal (home: nod 30s + auto-emotion), 2=full (host)
MOTION_EFFECTS_LEVEL: Final[int] = int(os.environ.get("MOTION_EFFECTS", "1"))
# Seconds of silence before active → dormant transition
CONVERSATION_ACTIVE_TIMEOUT: Final[float] = float(os.environ.get("CONVERSATION_ACTIVE_TIMEOUT", "30.0"))

# OpenAI voice → Gemini voice mapping
_OPENAI_TO_GEMINI_VOICE: Dict[str, str] = {
    "coral": "Kore",
    "alloy": "Aoede",
    "aria": "Leda",
    "ballad": "Fenrir",
    "sage": "Charon",
    "verse": "Puck",
    "cedar": "Zephyr",
}

GEMINI_VOICES: Final[List[str]] = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Aoede",
    "Leda", "Orus", "Autonoe", "Erinome", "Schedar", "Sadachbia",
    "Algieba", "Achernar", "Zubenelgenubi", "Callirrhoe", "Despina",
    "Alnilam", "Vindemiatrix", "Umbriel", "Laomedeia", "Achird",
    "Enceladus", "Algenib", "Gacrux", "Sadaltager", "Iapetus",
    "Rasalgethi", "Pulcherrima", "Sulafat",
]


def _convert_tools_to_gemini(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format tool specs to Gemini function_declarations.

    OpenAI format:  {"type": "function", "name": ..., "description": ..., "parameters": {...}}
    Gemini format:  {"function_declarations": [{"name": ..., "description": ..., "parameters": {...}}]}
    """
    declarations: List[Dict[str, Any]] = []
    for spec in tool_specs:
        decl: Dict[str, Any] = {
            "name": spec["name"],
            "description": spec.get("description", ""),
        }
        params = spec.get("parameters")
        if params and isinstance(params, dict) and params.get("properties"):
            decl["parameters"] = params
        declarations.append(decl)
    return [{"function_declarations": declarations}]


class GeminiLiveHandler(AsyncStreamHandler):
    """Gemini Live API handler implementing fastrtc AsyncStreamHandler.

    Compatible with both Gradio mode and headless LocalStream mode.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_INPUT_SAMPLE_RATE,  # Output at 16kHz (robot native)
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Session state
        self.session: Any = None
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        # Timing
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Lifecycle
        self._shutdown_requested = False
        self._reconnect_requested = False
        self._connected_event = asyncio.Event()
        self._receive_task: Optional[asyncio.Task[None]] = None

        # Echo suppression: mute mic input while robot is speaking
        self._model_speaking = False
        self._last_audio_data_time: float = 0.0
        self._last_echo_reset_log: float = 0.0  # rate-limit echo suppression log

        # Energy gate: filter out ambient/distant speech before sending to Gemini
        self._energy_gate_last_active: float = 0.0  # monotonic timestamp
        self._energy_gate_open: bool = False  # True while gate is open (speech detected)
        # Pre-buffer: ring buffer of recent audio frames for lookback on speech onset.
        # Max frames = prebuffer_sec / frame_duration. Each frame ≈ 20-40ms at 16kHz.
        from collections import deque
        self._prebuffer: deque[bytes] = deque(
            maxlen=max(1, int(ENERGY_GATE_PREBUFFER_SEC / 0.02))  # ~15 frames for 300ms
        )

        # Auto-emotion: one emotion per model turn, triggered by keyword in output text.
        # Deferred to post-turn: emotion is detected during speech but queued only after
        # turn_complete, preventing motor movement from disrupting audio playback.
        self._emotion_played_this_turn: bool = False
        self._pending_auto_emotion: str | None = None

        # Track whether Gemini has responded in this gate cycle (for thinking motion gating)
        self._gemini_responded_this_cycle: bool = False

        # Conversation mode: "dormant" (high threshold, no reactions) / "active" (low threshold, reactions)
        self._conversation_mode: str = "dormant"
        self._last_gemini_audio_time: float = 0.0

        # Music playback: when True, SBER gate is bypassed (music distorts speech ratio)
        self._music_playing: bool = False

        # Post-turn echo window: after model finishes speaking, speaker echo has
        # SBER ≈ 0.35 (below 0.5 threshold) and gets BLOCKED as "music".
        # During this window, SBER is bypassed so echo doesn't block real speech.
        self._post_turn_echo_until: float = 0.0

        # Listening nods: subtle micro-nods while actively listening (Feature A)
        self._last_listening_nod_time: float = 0.0
        self._listening_start_time: float = 0.0

        # Tool-call waiting animation state
        self._tool_call_in_progress: bool = False
        self._tool_call_start_time: float = 0.0
        self._tool_call_last_emotion: str = ""  # avoid repeating the same expression

        # Response generation micro-motion: keep alive between audio chunks
        self._last_response_micromove_time: float = 0.0

        # Audio buffer for resampling (accumulate small Gemini chunks before output)
        self._audio_buffer = bytearray()
        # Output at robot's native 16kHz to avoid per-chunk resampling artifacts
        self._output_sr = GEMINI_INPUT_SAMPLE_RATE  # 16000

        # Speaker verification (fail-open: disabled if not available)
        self._speaker_verified_this_session: bool = False
        self._speaker_verifier = None
        self._last_stranger_log: float = 0.0
        try:
            import sys
            sys.path.insert(0, os.path.expanduser("~/vicky_conversation/metis_extensions"))
            from speaker_verification.verifier import SpeakerVerifier
            _sv_enabled = os.environ.get("SPEAKER_VERIFY_ENABLED", "0") == "1"
            if _sv_enabled:
                self._speaker_verifier = SpeakerVerifier()
                logger.info("Speaker verification initialized")
        except Exception as e:
            logger.info("Speaker verification not available: %s", e)

    def _clear_model_speaking(self) -> None:
        """Reset echo suppression flag (called via call_later after turn_complete)."""
        if self._model_speaking:
            self._model_speaking = False
            # Start post-turn echo window: speaker echo decays over ~3s.
            # During this window, SBER gate is bypassed so echo (SBER≈0.35)
            # doesn't block subsequent real speech.
            import time as _time_cms
            self._post_turn_echo_until = _time_cms.monotonic() + 3.0
            logger.debug("Echo suppression cleared (post-turn delay) — echo window 3s")

    def _play_deferred_emotion(self, emotion_name: str) -> None:
        """Play a deferred auto-emotion after turn_complete + delay.

        Called via call_later to avoid motor movement during audio playback.
        """
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
            if not hasattr(self, '_emotion_moves'):
                self._emotion_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            self.deps.movement_manager.queue_move(EmotionQueueMove(emotion_name, self._emotion_moves))
            logger.debug("Auto-emotion played (post-turn): %s", emotion_name)
        except Exception as e:
            logger.debug("Auto-emotion skipped: %s", e)

    def copy(self) -> GeminiLiveHandler:
        """Create a copy of the handler (required by fastrtc for Gradio mode)."""
        return GeminiLiveHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    #  API key resolution
    # ------------------------------------------------------------------ #
    def _resolve_api_key(self) -> str:
        """Resolve Gemini API key from environment."""
        key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or getattr(config, "GEMINI_API_KEY", None)
            or ""
        )
        return key.strip()

    # ------------------------------------------------------------------ #
    #  Voice resolution (maps OpenAI voice names to Gemini)
    # ------------------------------------------------------------------ #
    def _resolve_voice(self) -> str:
        """Get the voice name for Gemini, mapping OpenAI names if needed."""
        voice_str = get_session_voice(default="cedar")
        mapped = _OPENAI_TO_GEMINI_VOICE.get(voice_str.lower())
        if mapped:
            return mapped
        # If the voice name is already a valid Gemini voice, use it directly
        if voice_str in GEMINI_VOICES:
            return voice_str
        # Fallback
        return "Kore"

    # ------------------------------------------------------------------ #
    #  Lifecycle: start_up / shutdown
    # ------------------------------------------------------------------ #
    async def start_up(self) -> None:
        """Connect to Gemini Live API with auto-reconnect on session expiry."""
        api_key = self._resolve_api_key()
        if not api_key:
            logger.error(
                "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment."
            )
            return

        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )

        # Outer reconnect loop (handles 15-min session limit)
        max_consecutive_failures = 5
        consecutive_failures = 0
        while not self._shutdown_requested:
            try:
                await self._run_session()
                consecutive_failures = 0  # Reset on successful session
                if self._shutdown_requested:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._shutdown_requested:
                    break
                consecutive_failures += 1
                logger.warning(
                    "Gemini session ended (%s). Failure %d/%d.",
                    e, consecutive_failures, max_consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Gemini: %d consecutive failures — stopping reconnect. "
                        "Check API key and network.",
                        max_consecutive_failures,
                    )
                    break
                self.session = None
                self._connected_event.clear()
                # Exponential backoff: 2s, 4s, 8s, 16s...
                backoff = min(2 ** consecutive_failures, 30)
                await asyncio.sleep(backoff)

    async def _run_session(self) -> None:
        """Establish and manage a single Gemini Live session."""
        instructions = get_session_instructions()
        voice = self._resolve_voice()
        tool_specs = get_tool_specs()
        gemini_tools = _convert_tools_to_gemini(tool_specs)

        model = os.environ.get("GEMINI_LIVE_MODEL", DEFAULT_GEMINI_MODEL)

        session_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                ),
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=instructions)]
            ),
            tools=gemini_tools,
        )

        logger.info(
            "Connecting to Gemini Live: model=%s voice=%s profile=%r",
            model,
            voice,
            getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
        )

        async with self.client.aio.live.connect(model=model, config=session_config) as session:
            self.session = session
            self._connected_event.set()
            self._reconnect_requested = False
            logger.info("Gemini Live session connected")

            # Start daemon health monitor (病根 2: detect Zenoh corruption)
            health_task = asyncio.create_task(self._daemon_health_monitor())

            try:
                await self._receive_loop()
            finally:
                health_task.cancel()
                self.session = None
                self._connected_event.clear()

    # ------------------------------------------------------------------ #
    #  Daemon health monitor (病根 2: Zenoh corruption recovery)
    # ------------------------------------------------------------------ #
    async def _daemon_health_monitor(self) -> None:
        """Periodically check robot daemon health via Zenoh SDK.

        When Zenoh is corrupted (ready=False, last_alive=None), the audio
        pipeline silently degrades. This monitor detects the bad state and
        triggers a graceful self-restart so the daemon restarts the app
        with a fresh Zenoh session.
        """
        HEALTH_CHECK_INTERVAL = 60.0  # seconds between checks
        UNHEALTHY_THRESHOLD = 120.0   # seconds before triggering restart
        unhealthy_since: float = 0.0

        await asyncio.sleep(15.0)  # let startup settle

        while not self._shutdown_requested:
            try:
                # SDK call goes through Zenoh — if Zenoh is broken, this times out
                loop = asyncio.get_event_loop()
                status = await asyncio.wait_for(
                    loop.run_in_executor(None, self.deps.reachy_mini.client.get_status),
                    timeout=10.0,
                )
                # Check for degraded state
                import time as _time_hm
                if not status or status.get("ready") is False:
                    if unhealthy_since == 0:
                        unhealthy_since = _time_hm.monotonic()
                        logger.warning("Daemon health: DEGRADED (ready=%s)", status.get("ready"))
                    elif _time_hm.monotonic() - unhealthy_since > UNHEALTHY_THRESHOLD:
                        logger.error(
                            "Daemon health: UNHEALTHY for >%.0fs — Zenoh likely corrupted. "
                            "Triggering graceful exit for daemon auto-restart.",
                            UNHEALTHY_THRESHOLD,
                        )
                        # Attempt daemon restart via HTTP API before exiting
                        await self._attempt_daemon_restart()
                        # Exit the process — daemon will auto-restart with fresh Zenoh
                        self._shutdown_requested = True
                        return
                else:
                    if unhealthy_since > 0:
                        logger.info("Daemon health: recovered")
                    unhealthy_since = 0

            except asyncio.TimeoutError:
                import time as _time_hm
                if unhealthy_since == 0:
                    unhealthy_since = _time_hm.monotonic()
                    logger.warning("Daemon health: Zenoh get_status timeout (10s)")
                elif _time_hm.monotonic() - unhealthy_since > UNHEALTHY_THRESHOLD:
                    logger.error(
                        "Daemon health: Zenoh unresponsive for >%.0fs — triggering restart.",
                        UNHEALTHY_THRESHOLD,
                    )
                    await self._attempt_daemon_restart()
                    self._shutdown_requested = True
                    return

            except Exception as e:
                logger.debug("Daemon health check exception: %s", e)

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _attempt_daemon_restart(self) -> None:
        """Try to restart the robot daemon via HTTP API."""
        import urllib.request
        robot_ip = os.environ.get("ROBOT_IP", "192.168.0.72")
        base = f"http://{robot_ip}:8000/api"
        for action, endpoint in [
            ("stop app", f"{base}/apps/stop-current-app"),
            ("stop daemon", f"{base}/daemon/stop"),
        ]:
            try:
                req = urllib.request.Request(endpoint, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info("Daemon recovery: %s → %s", action, resp.status)
            except Exception as e:
                logger.warning("Daemon recovery: %s failed: %s", action, e)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        self.session = None

        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------ #
    #  Receive loop: audio, tool calls, transcription from Gemini
    # ------------------------------------------------------------------ #
    async def _receive_loop(self) -> None:
        """Continuously receive from Gemini session and dispatch events."""
        while self.session and not self._shutdown_requested and not self._reconnect_requested:
            try:
                async for response in self.session.receive():
                    if self._shutdown_requested or self._reconnect_requested:
                        break

                    # --- Audio data ---
                    if response.data is not None:
                        import time as _time
                        self._model_speaking = True
                        self._last_audio_data_time = _time.time()
                        self._last_gemini_audio_time = _time.monotonic()
                        self.last_activity_time = asyncio.get_event_loop().time()

                        # Feed head wobbler for audio-reactive motion
                        if self.deps.head_wobbler is not None:
                            import base64
                            self.deps.head_wobbler.feed(
                                base64.b64encode(response.data).decode()
                            )

                        # --- Magic Trick 3: micro-motion during response generation ---
                        # While Gemini streams audio (before turn_complete), queue a
                        # subtle expression every ~5s so the robot doesn't freeze
                        # between head-wobble pauses. Lightweight: one random.choice.
                        now_mono = _time.monotonic()
                        if MOTION_EFFECTS_LEVEL >= 2 and now_mono - self._last_response_micromove_time > 5.0:
                            self._last_response_micromove_time = now_mono
                            _micro_emotions = ["calming1", "understanding2"]
                            _pick = random.choice(_micro_emotions)
                            try:
                                from reachy_mini.motion.recorded_move import RecordedMoves
                                from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
                                if not hasattr(self, "_micromove_moves"):
                                    self._micromove_moves = RecordedMoves(
                                        "pollen-robotics/reachy-mini-emotions-library"
                                    )
                                self.deps.movement_manager.queue_move(
                                    EmotionQueueMove(_pick, self._micromove_moves)
                                )
                                logger.debug("Response micro-motion: %s", _pick)
                            except Exception:
                                pass

                        # Buffer small chunks, then resample as a batch to avoid
                        # per-chunk resampling artifacts (Gemini sends ~40ms chunks)
                        self._audio_buffer.extend(response.data)

                        # Flush when we have >= 80ms of audio (4800 samples @ 24kHz)
                        min_bytes = 1920 * 2  # 200ms @ 24kHz, 16-bit
                        if len(self._audio_buffer) >= min_bytes:
                            await self._flush_audio_buffer()

                    # --- Text response ---
                    if response.text is not None:
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {"role": "assistant", "content": response.text}
                            )
                        )

                    # --- Tool calls ---
                    if hasattr(response, "tool_call") and response.tool_call:
                        await self._handle_tool_calls(response.tool_call)

                    # --- Server content (transcription, interruption) ---
                    if response.server_content is not None:
                        await self._handle_server_content(response.server_content)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._shutdown_requested:
                    break
                logger.warning("Receive loop error: %s", e)
                # Session likely expired — exit to trigger reconnect
                break

    # ------------------------------------------------------------------ #
    #  Auto-emotion: keyword → emotion mapping (lightweight, no ML)
    # ------------------------------------------------------------------ #
    _EMOTION_KEYWORDS: list[tuple[list[str], str]] = [
        (["歡迎", "你好", "welcome", "hello"], "welcoming1"),
        (["太棒", "成功", "恭喜", "congratulations", "太好了"], "enthusiastic1"),
        (["不確定", "可能", "perhaps", "我想想"], "thoughtful1"),
        (["對不起", "抱歉", "sorry", "不好意思"], "understanding1"),
        (["再見", "掰掰", "goodbye", "下次見"], "loving1"),
        (["哈哈", "好笑", "funny"], "laughing1"),
        (["驚訝", "真的嗎", "wow", "天啊"], "surprised1"),
    ]

    def _classify_quick_emotion(self, text: str) -> str | None:
        """Ultra-lightweight keyword matching for auto-emotion.

        Returns an emotion name if any keyword is found in text, else None.
        First match wins. No match = neutral (no emotion played).
        """
        lower = text.lower()
        for keywords, emotion in self._EMOTION_KEYWORDS:
            for kw in keywords:
                if kw in lower:
                    return emotion
        return None

    async def _handle_server_content(self, sc: Any) -> None:
        """Process Gemini server_content for transcription and interruption."""
        _profile = os.environ.get("REACHY_MINI_CUSTOM_PROFILE", "default")
        _mode = self._conversation_mode

        # Input transcription (user speech → text) — new user turn resets emotion flag
        if hasattr(sc, "input_transcription") and sc.input_transcription:
            text = getattr(sc.input_transcription, "text", None)
            if text:
                self._emotion_played_this_turn = False
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": text})
                )
                if _transcript_writer is not None:
                    try:
                        _transcript_writer.write_turn(
                            "user", text, profile=_profile, conv_mode=_mode
                        )
                    except Exception as e:
                        logger.debug("transcript write (user) failed: %s", e)

        # Output transcription (model speech → text)
        if hasattr(sc, "output_transcription") and sc.output_transcription:
            text = getattr(sc.output_transcription, "text", None)
            if text:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": text})
                )
                if _transcript_writer is not None:
                    try:
                        _transcript_writer.write_turn(
                            "assistant", text, profile=_profile, conv_mode=_mode
                        )
                    except Exception as e:
                        logger.debug("transcript write (assistant) failed: %s", e)
                # Auto-emotion: detect emotion keyword but DEFER playback to turn_complete.
                # Playing emotions during speech causes motor movement that can disrupt
                # the audio pipeline (observed: "Cleared player queue" during playback).
                if text and not self._emotion_played_this_turn:
                    emotion = self._classify_quick_emotion(text)
                    if emotion:
                        self._pending_auto_emotion = emotion
                        self._emotion_played_this_turn = True
                        logger.debug("Auto-emotion detected (deferred): %s", emotion)

        # Interruption (user started speaking while model was talking)
        if hasattr(sc, "interrupted") and sc.interrupted:
            # self._model_speaking = False  # DISABLED: let echo suppression continue
            # self._audio_buffer.clear()  # DISABLED: echo causes false interruption
            logger.debug("Model interrupted by user speech")
            # DISABLED: echo causes false server interruption; barge-in handles real ones
            if False and hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            if self._conversation_mode == "active":
                self.deps.movement_manager.set_listening(True)
                self.deps.movement_manager.trigger_listening_reaction()

        # Turn complete — flush remaining audio buffer and re-enable mic
        if hasattr(sc, "turn_complete") and sc.turn_complete:
            if self._audio_buffer:
                await self._flush_audio_buffer()
            # Delayed reset: give speaker 2s to finish playing buffered audio,
            # then clear echo suppression so mic can hear again.
            asyncio.get_event_loop().call_later(2.0, self._clear_model_speaking)
            # Reset micro-motion timer so next turn starts fresh
            self._last_response_micromove_time = 0.0
            self._gemini_responded_this_cycle = True
            if self._conversation_mode == "dormant":
                self._conversation_mode = "active"
                import time as _time_tc
                self._last_gemini_audio_time = _time_tc.monotonic()  # reset so timeout counts from now
                logger.info("Conversation mode: DORMANT -> ACTIVE")
            # Play deferred auto-emotion AFTER turn completes (speech done).
            # Delay 2.5s to let speaker finish playing buffered audio first.
            if self._pending_auto_emotion:
                _deferred_emotion = self._pending_auto_emotion
                self._pending_auto_emotion = None
                asyncio.get_event_loop().call_later(
                    2.5, self._play_deferred_emotion, _deferred_emotion
                )
            logger.debug("Gemini turn complete")

    # ------------------------------------------------------------------ #
    #  Audio buffer flush: resample 24kHz → 16kHz in batch
    # ------------------------------------------------------------------ #
    async def _flush_audio_buffer(self) -> None:
        """Resample buffered 24kHz audio to 16kHz and push to output queue."""
        if not self._audio_buffer:
            return

        raw = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        audio_24k = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Apply volume gain
        if VOLUME_GAIN != 1.0:
            audio_24k = audio_24k * VOLUME_GAIN

        # Resample 24kHz → 16kHz in one batch (avoids per-chunk artifacts)
        num_samples_16k = int(len(audio_24k) * self._output_sr / GEMINI_OUTPUT_SAMPLE_RATE)
        audio_16k = scipy_resample(audio_24k, num_samples_16k)

        # Clip and convert back to int16
        audio_out = np.clip(audio_16k, -32768, 32767).astype(np.int16)

        await self.output_queue.put((self._output_sr, audio_out))

    # ------------------------------------------------------------------ #
    #  Tool call handling (uses existing core_tools dispatch)
    # ------------------------------------------------------------------ #
    # Emotion pools for tool-call waiting animation (Magic Trick)
    _TOOL_WAIT_EMOTIONS_IMMEDIATE: Final[List[str]] = ["thoughtful1", "curious1"]
    _TOOL_WAIT_EMOTIONS_EXTENDED: Final[List[str]] = ["understanding1", "attentive1"]

    def _queue_tool_wait_emotion(self, pool: List[str]) -> str:
        """Queue a random emotion from pool, avoiding the last-played one.

        Returns the emotion name that was queued (for dedup tracking).
        Lightweight: no ML, just random.choice with exclusion.
        """
        candidates = [e for e in pool if e != self._tool_call_last_emotion]
        if not candidates:
            candidates = list(pool)  # fallback: allow repeat if pool is tiny
        emotion = random.choice(candidates)
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
            if not hasattr(self, "_tool_wait_moves"):
                self._tool_wait_moves = RecordedMoves(
                    "pollen-robotics/reachy-mini-emotions-library"
                )
            self.deps.movement_manager.queue_move(
                EmotionQueueMove(emotion, self._tool_wait_moves)
            )
            logger.debug("Tool-wait emotion: %s", emotion)
        except Exception as e:
            logger.debug("Tool-wait emotion skipped: %s", e)
        return emotion

    async def _handle_tool_calls(self, tool_call: Any) -> None:
        """Process Gemini tool calls via existing dispatch_tool_call system.

        Magic Trick: queue expressive motions so the robot looks alive
        while waiting for tool results (3-15s typical latency).
        """
        import time as _time

        function_responses: List[types.FunctionResponse] = []
        self._tool_call_in_progress = True

        for fc in tool_call.function_calls:
            tool_name = fc.name
            args_dict = dict(fc.args) if fc.args else {}
            args_json = json.dumps(args_dict)

            logger.info("Tool call: %s(%s)", tool_name, args_json)

            # --- Magic Trick 1: Immediate "thinking" expression ---
            self._tool_call_start_time = _time.monotonic()
            if MOTION_EFFECTS_LEVEL >= 2:
                self._tool_call_last_emotion = self._queue_tool_wait_emotion(
                    self._TOOL_WAIT_EMOTIONS_IMMEDIATE
                )

            # --- Dispatch with extended-wait animation ---
            # Run tool dispatch concurrently with a watcher that fires
            # a second expression if the call takes > 3 seconds.
            async def _extended_wait_watcher() -> None:
                """Magic Trick 2: after 3s, queue a second distinct expression."""
                await asyncio.sleep(3.0)
                if self._tool_call_in_progress and MOTION_EFFECTS_LEVEL >= 2:
                    self._tool_call_last_emotion = self._queue_tool_wait_emotion(
                        self._TOOL_WAIT_EMOTIONS_EXTENDED
                    )

            watcher_task = asyncio.create_task(_extended_wait_watcher())

            try:
                tool_result = await dispatch_tool_call(tool_name, args_json, self.deps)
                logger.debug("Tool '%s' result: %s", tool_name, tool_result)
            except Exception as e:
                logger.error("Tool '%s' failed: %s", tool_name, e)
                tool_result = {"error": str(e)}
            finally:
                watcher_task.cancel()
                try:
                    await watcher_task
                except asyncio.CancelledError:
                    pass

            # Persist tool call to voice memory L1 (JSONL hot log)
            if _transcript_writer is not None:
                try:
                    _transcript_writer.write_turn(
                        "tool",
                        json.dumps({"name": tool_name, "args": args_dict, "result": tool_result},
                                    ensure_ascii=False),
                        tool_calls=[{"name": tool_name, "args": args_dict, "result": tool_result}],
                        profile=os.environ.get("REACHY_MINI_CUSTOM_PROFILE", "default"),
                        conv_mode=self._conversation_mode,
                    )
                except Exception as _e:
                    logger.debug("transcript write (tool) failed: %s", _e)

            # Emit tool result to chatbot UI
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant",
                    "content": json.dumps(tool_result),
                    "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                })
            )

            # Music playback flag: bypass SBER when music is playing
            if tool_name == "play_music":
                status = tool_result.get("status", "")
                if status == "playing":
                    self._music_playing = True
                    logger.info("Music started — SBER gate bypassed")
                elif status in ("stopped", "not_playing"):
                    self._music_playing = False
                    logger.info("Music stopped — SBER gate restored")

            # Camera tool: send captured image to Gemini for visual understanding
            if tool_name == "camera" and "b64_im" in tool_result:
                import base64 as b64mod
                try:
                    img_bytes = b64mod.b64decode(tool_result["b64_im"])
                    if self.session:
                        await self.session.send_realtime_input(
                            video=types.Blob(data=img_bytes, mime_type="image/jpeg")
                        )
                        logger.info("Sent camera image to Gemini session")
                except Exception as e:
                    logger.warning("Failed to send camera image: %s", e)

                # Emit image to Gradio chatbot
                if self.deps.camera_worker is not None:
                    try:
                        import cv2
                        import gradio as gr
                        np_img = self.deps.camera_worker.get_latest_frame()
                        if np_img is not None:
                            rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_frame = None
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {"role": "assistant", "content": gr.Image(value=rgb_frame)}
                            )
                        )
                    except Exception:
                        pass

            # Build function response for Gemini
            response_payload = tool_result if isinstance(tool_result, dict) else {"result": str(tool_result)}
            function_responses.append(
                types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response=response_payload,
                )
            )

        self._tool_call_in_progress = False

        # Send all tool responses back to Gemini
        if self.session and function_responses:
            await self.session.send_tool_response(
                function_responses=function_responses,
            )

        # Reset head wobbler after tool execution
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    # ------------------------------------------------------------------ #
    #  Audio I/O: receive() and emit()
    # ------------------------------------------------------------------ #
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from mic and send to Gemini.

        Args:
            frame: (sample_rate, audio_data) tuple from microphone.

        """
        if not self.session:
            return

        # Echo suppression: skip mic input while robot is speaking
        # Safety: auto-reset after 3s of no new audio from Gemini
        if self._model_speaking:
            import time as _time
            if _time.time() - self._last_audio_data_time > 6.0:
                self._model_speaking = False
                logger.debug("Echo suppression auto-reset (6s timeout)")
            else:
                return

        input_sample_rate, audio_frame = frame

        # Reshape: ensure 1D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
            else:
                audio_frame = audio_frame.ravel()

        # Resample to 16 kHz if needed
        if input_sample_rate != GEMINI_INPUT_SAMPLE_RATE:
            num_samples = int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sample_rate)
            audio_frame = scipy_resample(audio_frame, num_samples)

        # Ensure int16
        audio_frame = audio_to_int16(audio_frame)

        # Speech gate: two-factor authentication for human speech.
        # Factor 1: RMS energy (volume) — filters very quiet ambient noise.
        # Factor 2: Speech Band Energy Ratio (SBER) — filters music/TV.
        #   Human speech concentrates 60-80% energy in 300-3400 Hz.
        #   Music/noise spreads energy across full spectrum (ratio < 0.4).
        # Audio must pass BOTH to reach Gemini.
        pcm_bytes = audio_frame.tobytes()

        if ENERGY_GATE_THRESHOLD > 0:
            import time as _time
            audio_f32 = audio_frame.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
            now = _time.monotonic()

            # Speech band energy ratio via FFT
            speech_ratio = -1.0
            is_speech_like = True  # default pass if SBER disabled
            # Lower SBER threshold during post-turn echo window.
            # Speaker echo has SBER ≈ 0.35 — fully bypassing SBER (0.0) causes echo
            # feedback loops. Lowering to 0.40 blocks pure echo while letting through
            # real speech (SBER ≥ 0.5) or speech mixed with residual echo (SBER ~0.4-0.5).
            _in_echo_window = (self._post_turn_echo_until > 0 and now < self._post_turn_echo_until)
            if self._music_playing:
                _sber_threshold = 0.0
            elif _in_echo_window:
                _sber_threshold = 0.40
            else:
                _sber_threshold = SPEECH_BAND_RATIO_THRESHOLD
            if _sber_threshold > 0 and len(audio_f32) >= 256:
                fft = np.fft.rfft(audio_f32)
                power = np.abs(fft) ** 2
                freqs = np.fft.rfftfreq(len(audio_f32), 1.0 / GEMINI_INPUT_SAMPLE_RATE)
                speech_mask = (freqs >= 300) & (freqs <= 3400)
                total_power = float(np.sum(power)) + 1e-10
                speech_ratio = float(np.sum(power[speech_mask])) / total_power
                is_speech_like = speech_ratio >= _sber_threshold

            # Dynamic threshold based on conversation mode.
            # During echo window, force dormant threshold to filter speaker echo.
            if self._conversation_mode == "dormant" or _in_echo_window:
                _effective_threshold = ENERGY_GATE_DORMANT
            else:
                _effective_threshold = ENERGY_GATE_THRESHOLD

            if rms >= _effective_threshold and is_speech_like:
                # Speech detected (loud enough + right spectral shape)
                if not self._energy_gate_open:
                    # Speaker verification: dormant=fail-close, active=fail-open
                    _fail_open = (self._conversation_mode == "active")
                    if self._speaker_verifier and not self._speaker_verified_this_session:
                        _audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                        _audio_f32 = _audio_i16.astype(np.float32) / 32768.0
                        is_ian = self._speaker_verifier.verify(
                            _audio_f32, fail_open=_fail_open)
                        if not is_ian:
                            import time as _time_sv
                            _now_sv = _time_sv.monotonic()
                            if _now_sv - self._last_stranger_log > 10.0:
                                self._last_stranger_log = _now_sv
                                logger.debug("BLOCKED (stranger) RMS=%.4f SBER=%.2f mode=%s",
                                             rms, speech_ratio, self._conversation_mode)
                            self._prebuffer.append(pcm_bytes)
                            return
                        else:
                            self._speaker_verified_this_session = True
                            logger.info("Speaker verified: Ian")
                    self._energy_gate_open = True
                    self._listening_start_time = now
                    self._last_listening_nod_time = now
                    # Physical reactions only in active mode
                    if self._conversation_mode == "active":
                        self.deps.movement_manager.set_listening(True)
                        self.deps.movement_manager.trigger_listening_reaction()
                        if rms >= _effective_threshold * 3:
                            try:
                                from reachy_mini.motion.recorded_move import RecordedMoves
                                from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
                                if not hasattr(self, '_attention_moves'):
                                    self._attention_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                                self.deps.movement_manager.queue_move(
                                    EmotionQueueMove("attentive1", self._attention_moves)
                                )
                                logger.debug("Far attention reaction (RMS=%.3f)", rms)
                            except Exception:
                                pass
                    for buffered in self._prebuffer:
                        try:
                            await self.session.send_realtime_input(
                                audio=types.Blob(data=buffered, mime_type="audio/pcm;rate=16000")
                            )
                        except Exception:
                            pass
                    self._prebuffer.clear()
                    logger.debug("Speech gate OPEN (RMS=%.4f, SBER=%.2f, mode=%s, thr=%.3f)",
                                 rms, speech_ratio, self._conversation_mode, _effective_threshold)
                self._energy_gate_last_active = now
            else:
                # Blocked: too quiet or wrong spectral shape
                if self._energy_gate_open:
                    if now - self._energy_gate_last_active > ENERGY_GATE_HOLD_SEC:
                        self._energy_gate_open = False
                        if self._speaker_verifier:
                            self._speaker_verifier.reset()
                        self.deps.movement_manager.set_listening(False)
                        self._prebuffer.clear()
                        logger.debug("Speech gate CLOSED (silence %.1fs)", now - self._energy_gate_last_active)
                        # Active → Dormant check on gate close
                        if self._conversation_mode == "active":
                            time_since_gemini = now - self._last_gemini_audio_time
                            if time_since_gemini >= CONVERSATION_ACTIVE_TIMEOUT:
                                self._conversation_mode = "dormant"
                                self._speaker_verified_this_session = False
                                self._gemini_responded_this_cycle = False
                                logger.info("Conversation mode: ACTIVE -> DORMANT (%.0fs silence)",
                                            time_since_gemini)
                        else:
                            # Dormant: reset verification on every gate close
                            self._speaker_verified_this_session = False
                            self._gemini_responded_this_cycle = False
                        # Thinking motion: only in active mode after Gemini responded
                        if MOTION_EFFECTS_LEVEL >= 2 and self._conversation_mode == "active" and self._gemini_responded_this_cycle:
                            try:
                                from reachy_mini.motion.recorded_move import RecordedMoves
                                from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
                                if not hasattr(self, "_thinking_moves"):
                                    self._thinking_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                                self.deps.movement_manager.queue_move(
                                    EmotionQueueMove("thoughtful1", self._thinking_moves)
                                )
                                logger.debug("Queued thinking motion")
                            except Exception as e:
                                logger.debug("Thinking motion skipped: %s", e)
                        self._gemini_responded_this_cycle = False
                        return
                    # Still in hold period → forward audio (mid-sentence pause)
                else:
                    # Active → Dormant timeout check (while gate is closed)
                    if (self._conversation_mode == "active"
                            and self._last_gemini_audio_time > 0
                            and now - self._last_gemini_audio_time >= CONVERSATION_ACTIVE_TIMEOUT):
                        self._conversation_mode = "dormant"
                        self._speaker_verified_this_session = False
                        logger.info("Conversation mode: ACTIVE -> DORMANT (timeout while idle)")
                    # Log periodically (every ~5s)
                    if not hasattr(self, '_last_block_log') or now - self._last_block_log > 5.0:
                        self._last_block_log = now
                        reason = "quiet" if rms < _effective_threshold else "music"
                        logger.debug("BLOCKED (%s) RMS=%.4f SBER=%.2f mode=%s",
                                     reason, rms, speech_ratio, self._conversation_mode)
                    self._prebuffer.append(pcm_bytes)
                    return
        # Send raw PCM bytes to Gemini (3.1: audio Blob, not media_chunks)
        try:
            await self.session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
            )
        except Exception:
            # Session may be closing/reconnecting — drop frame silently
            pass

        # Periodic listening nods: only in active mode, level 1+=slow(25-35s), level 2=fast(8-15s)
        if MOTION_EFFECTS_LEVEL >= 1 and self._conversation_mode == "active" and self._energy_gate_open and ENERGY_GATE_THRESHOLD > 0:
            import time as _time
            now = _time.monotonic()
            listening_duration = now - self._listening_start_time
            time_since_nod = now - self._last_listening_nod_time
            _nod_interval = random.uniform(25.0, 35.0) if MOTION_EFFECTS_LEVEL == 1 else random.uniform(8.0, 15.0)
            if listening_duration > 5.0 and time_since_nod > _nod_interval:
                try:
                    from reachy_mini.motion.recorded_move import RecordedMoves
                    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
                    if not hasattr(self, '_nod_moves'):
                        self._nod_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                    # Use understanding2 for a subtle nod
                    self.deps.movement_manager.queue_move(
                        EmotionQueueMove("understanding2", self._nod_moves)
                    )
                    self._last_listening_nod_time = now
                    logger.debug("Listening nod")
                except Exception:
                    pass

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio/outputs to speaker and chatbot."""
        # Idle behavior — DISABLED (2026-02-22)
        # Gemini Live native audio ignores system_instruction idle directives.
        # send_idle_signal() sends "express yourself" as user message, which
        # overrides profile instructions and causes unprompted motivational speech.
        # Fix: stop sending idle signals entirely. Robot stays silent when idle.

        return await wait_for_item(self.output_queue)

    # ------------------------------------------------------------------ #
    #  Personality
    # ------------------------------------------------------------------ #
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality by triggering session reconnect."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            logger.info("Set profile to %r, requesting reconnect", profile)
            self._reconnect_requested = True
            return "Applied personality. Reconnecting with new instructions..."
        except Exception as e:
            logger.error("apply_personality failed: %s", e)
            return f"Failed to apply personality: {e}"

    # ------------------------------------------------------------------ #
    #  Idle signal
    # ------------------------------------------------------------------ #
    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send idle signal to Gemini to trigger spontaneous behavior.

        NOTE (2026-04): This method is DISABLED at the call site (see emit()).
        Gemini 3.1 Flash Live restricts send_client_content to initial context
        seeding only — use send_realtime_input for mid-session text if re-enabling.
        """
        if not self.session:
            return
        self.is_idle_tool_call = True
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        timestamp = f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"
        msg = (
            f"{timestamp} Idle for {idle_duration:.0f}s. "
            "Feel free to express yourself — dance, show an emotion, look around, or just be yourself!"
        )
        await self.session.send_client_content(
            turns=types.Content(parts=[types.Part(text=msg)]),
            turn_complete=True,
        )

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return available Gemini voices."""
        return list(GEMINI_VOICES)
