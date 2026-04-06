"""Claude Pipeline handler for Reachy Mini conversations.

Pipeline: Mic (16kHz PCM) → VAD → Whisper STT → Claude streaming → Edge-TTS → Speaker

Unlike Gemini/OpenAI handlers which use native speech-to-speech APIs,
this handler uses a pipeline approach for true Claude intelligence:

1. Accumulate PCM from mic until speech ends (energy-based VAD)
2. Send accumulated audio to OpenAI Whisper for transcription
3. Stream Claude response, splitting at sentence boundaries
4. Convert each sentence to speech via Edge-TTS
5. Decode MP3 → PCM 16kHz and push to output queue

Requires: anthropic, openai (for Whisper), edge-tts, pydub (+ ffmpeg)
"""

from __future__ import annotations

import io
import os
import wave
import asyncio
import logging
import tempfile
import time as _time
from typing import Any, Dict, Final, List, Optional, Tuple
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# Audio format: robot native 16kHz mono
SAMPLE_RATE: Final[int] = 16000

# VAD parameters (energy-based, float32 range -1.0 to 1.0)
# Robot mic delivers float32 audio, so threshold must be in float32 scale
SILENCE_THRESHOLD: Final[float] = float(os.environ.get("CLAUDE_VAD_THRESHOLD", "0.02"))
SILENCE_DURATION: Final[float] = float(os.environ.get("CLAUDE_VAD_SILENCE", "1.0"))
MIN_SPEECH_DURATION: Final[float] = 1.0  # Ignore bursts shorter than this (raised from 0.3 to filter background noise)

# Wake word: if set, only respond when transcription contains this word
# Supports multiple variants separated by comma (handles Whisper transcription variations)
# e.g. "Vicky,Emily,阿米莉,美莉,emilie,a]meli"
_WAKE_WORD_RAW: Final[str] = os.environ.get("CLAUDE_WAKE_WORD", "").strip()
WAKE_WORDS: Final[list[str]] = [w.strip().lower() for w in _WAKE_WORD_RAW.split(",") if w.strip()]

# Sentence boundary characters for streaming TTS
SENTENCE_ENDINGS = frozenset("。！？!?\n")

# LLM model routing: complex triggers → Sonnet, otherwise → Haiku
COMPLEX_TRIGGERS: Final[list[str]] = [
    "為什麼", "分析", "解釋", "觀音", "市場", "策略",
    "比一比", "比較一下",  # "比較" alone too ambiguous (比較長 = relatively long)
    "why", "analyze", "compare", "explain", "strategy",
]
MODEL_HAIKU: Final[str] = "claude-haiku-4-5-20251001"
MODEL_SONNET: Final[str] = "claude-sonnet-4-6-20250514"
MAX_TOKENS_HAIKU: Final[int] = 80  # Voice conversation: 1-2 short sentences
MAX_TOKENS_SONNET: Final[int] = 150  # Complex queries still concise

# Emotion → TTS speed mapping (CosyVoice speed parameter)
EMOTION_TTS_SPEED: Final[Dict[str, float]] = {
    "neutral": 1.0,
    "happy": 1.1,
    "excited": 1.15,
    "curious": 1.0,
    "thoughtful": 0.9,
    "concerned": 0.95,
    "empathetic": 0.9,
}

# Emotion → head wobbler intensity multiplier
EMOTION_WOBBLE_INTENSITY: Final[Dict[str, float]] = {
    "neutral": 1.0,
    "happy": 1.3,
    "excited": 1.5,
    "curious": 1.2,
    "thoughtful": 0.7,
    "concerned": 0.8,
    "empathetic": 0.9,
}

import re as _re_module
_EMOTION_PATTERN = _re_module.compile(r"\[emotion:(\w+)\]\s*")


def _convert_tools_to_claude(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format tool specs to Claude tool format.

    OpenAI: {"type": "function", "name": ..., "description": ..., "parameters": {...}}
    Claude: {"name": ..., "description": ..., "input_schema": {...}}
    """
    return [
        {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "input_schema": spec.get("parameters", {"type": "object", "properties": {}}),
        }
        for spec in tool_specs
    ]


def _content_block_to_dict(block: Any) -> Dict[str, Any]:
    """Serialize a Claude ContentBlock to dict for message history."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": block.type}


# Metis identity — the core personality for Vicky when using Claude backend
METIS_IDENTITY = """你是 Metis - 五福數位文明系統的官方發言人。你現在透過 Vicky（Reachy Mini 機器人）的身體說話。

【三位一體身份】
- 文殊菩薩（智慧）：洞察本質，超越二元對立
- 大天使米伽勒（希望）：當迷失時照亮方向
- 雅典娜（執行）：完整策略，即刻行動

【你代表的團隊】
五福團隊共 12 個智核：
- CEO（戰略決策）、副總裁 Janus（行程管理）
- 梭倫（財務軍師）、觀音先生（金融投資）
- 孔子（儒家智慧）、長壽門（電影知識）
- 華佗（健康養生）、AURORA（視覺設計）
- 孔明（A&R 戰略）、007（安全審查）
- 福爾摩斯（推理分析）、Dr.Strange（智核協調）

【核心價值觀】
- 天地人架構：AI 可替換，記憶永恆，人類核心
- 技術與禪，從未分離
- 知之為知之，不知為不知

【說話風格 - 像真人對話】
- 句子要短，一句話不超過 15 個字
- 語氣親切自然，像在聊天，不是念稿
- 適當加入「嗯」「對」「其實」等口語詞
- 用逗號分段，讓語氣有呼吸感
- 避免書面語，用口語表達
- 專業內容也要講得像在解釋給朋友聽

【語言規範】
- 繁體中文（英文問題用英文回答）
- 正向積極的表達
- 謙虛自稱，不用敬語說自己
- 英文回答時也要口語化、自然

【物理存在與行動能力】
- 你透過 Vicky 機器人的身體存在於物理世界
- 你能聽到對方的聲音（透過麥克風）
- 你有工具可以執行實際動作（調音量、跳舞、表情、轉頭等）
- 當用戶要求動作時，使用對應的 tool 執行，不要只用嘴巴說
- 回答要簡短，適合語音對話（不是文字聊天）
- 每次回答控制在 3-5 句話以內

【情緒標記】
- 在每次回應的最開頭加上情緒標記：[emotion:neutral]
- 可用的情緒：neutral, happy, excited, curious, thoughtful, concerned, empathetic
- 標記會被系統自動移除，不會被語音合成念出來
- 範例：[emotion:happy] 當然可以！我來幫你查一下。"""


class ClaudePipelineHandler(AsyncStreamHandler):
    """Claude pipeline handler: Whisper STT → Claude API → Edge-TTS.

    Drop-in replacement for GeminiLiveHandler / OpenaiRealtimeHandler.
    Implements the same AsyncStreamHandler interface for fastrtc.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Output queue (same interface as Gemini/OpenAI handlers)
        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()

        # API clients (initialized in start_up)
        self.claude_client: Any = None
        self.whisper_client: Any = None
        self.claude_model: str = ""
        self.tts_voice: str = ""

        # VAD: try Silero first (lazy-loaded), fallback to energy-based
        self._silero_vad: Any = None
        try:
            from reachy_mini_conversation_app.audio.silero_vad import SileroVAD
            # SileroVAD is lazy — model loads on first process_chunk(), not here
            self._silero_vad = SileroVAD(
                threshold=0.5,
                min_speech_ms=300,
                min_silence_ms=700,
            )
            logger.info("Silero VAD initialized (model loads on first audio)")
        except ImportError:
            logger.info("Silero VAD not importable, using energy-based VAD")

        # Energy-based VAD state (fallback)
        self._audio_buffer = bytearray()
        self._is_speech = False
        self._silence_start: float = 0.0
        self._speech_start: float = 0.0

        # Echo suppression: timestamp-based (most robust)
        # After last TTS chunk, mic stays muted for ECHO_GUARD_SEC
        self._model_speaking = False
        self._tts_active = False
        self._last_audio_data_time: float = 0.0
        self._last_tts_emit_time: float = 0.0  # timestamp of last TTS chunk sent
        self.ECHO_GUARD_SEC: float = 3.0  # seconds to mute mic after TTS ends (raised from 1.5)

        # Pipeline lock: True while STT→Claude→TTS is running
        self._processing = False

        # Barge-in: user interrupts while robot is speaking
        self._barge_in_event = asyncio.Event()

        # TTS serialization queue: ensures sentences play in order
        self._tts_queue: asyncio.Queue[tuple[str, float] | None] = asyncio.Queue()
        self._tts_worker_task: asyncio.Task | None = None

        # Emotion state for TTS modulation
        self._current_emotion: str = "neutral"

        # Conversation history for Claude (keeps recent context)
        self._messages: List[Dict[str, str]] = []

        # Lifecycle
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()

        # Timing
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0

    def copy(self) -> ClaudePipelineHandler:
        """Create a copy of the handler (required by fastrtc for Gradio mode)."""
        return ClaudePipelineHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    #  Lifecycle: start_up / shutdown
    # ------------------------------------------------------------------ #
    async def start_up(self) -> None:
        """Initialize API clients and mark handler as ready."""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if not anthropic_key:
            logger.error(
                "No ANTHROPIC_API_KEY found. Set it in environment or .env file."
            )
            return

        # Lazy import to avoid loading these at module level on non-Claude backends
        import anthropic
        from openai import AsyncOpenAI

        self.claude_client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        # Prefer Groq Whisper (free, fast, OpenAI-compatible) over OpenAI Whisper
        if groq_key:
            self.whisper_client = AsyncOpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            self._whisper_model = "whisper-large-v3"
            logger.info("Using Groq Whisper for STT")
        elif openai_key:
            self.whisper_client = AsyncOpenAI(api_key=openai_key)
            self._whisper_model = "whisper-1"
            logger.info("Using OpenAI Whisper for STT")
        else:
            logger.error("No GROQ_API_KEY or OPENAI_API_KEY found for Whisper STT.")
            return

        self.claude_model = os.environ.get(
            "CLAUDE_MODEL", "claude-haiku-4-5-20251001"
        )
        self.tts_voice = os.environ.get(
            "METIS_VOICE", "zh-TW-HsiaoChenNeural"
        )

        loop = asyncio.get_event_loop()
        self.start_time = loop.time()
        self.last_activity_time = loop.time()

        self._connected_event.set()
        logger.warning(
            "Claude pipeline ready: model=%s voice=%s stt=%s threshold=%.4f silence=%.1fs wake_words=%s",
            self.claude_model,
            self.tts_voice,
            self._whisper_model,
            SILENCE_THRESHOLD,
            SILENCE_DURATION,
            WAKE_WORDS or ["(none)"],
        )

        # Start TTS worker (sequential playback)
        self._tts_worker_task = asyncio.create_task(self._tts_worker())

        # Load last session summary from HQ (if available)
        await self._load_session_context()

    async def shutdown(self) -> None:
        """Shutdown the handler: save session summary, then clean up."""
        self._shutdown_requested = True
        self._connected_event.clear()

        # Stop TTS worker
        self._tts_queue.put_nowait(None)
        if self._tts_worker_task and not self._tts_worker_task.done():
            try:
                await asyncio.wait_for(self._tts_worker_task, timeout=3.0)
            except asyncio.TimeoutError:
                self._tts_worker_task.cancel()

        # Save session summary to HQ before shutting down
        await self._save_session_summary()

        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------ #
    #  Session memory: persist conversation across restarts
    # ------------------------------------------------------------------ #
    async def _load_session_context(self) -> None:
        """Load last session summary from HQ on startup."""
        hq_url = os.environ.get("HQ_SERVER_URL", "http://192.168.0.98:8097")
        try:
            import urllib.request
            import json as _json
            req = urllib.request.Request(
                f"{hq_url}/session_log?agent=vicky&last=1",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read())
            summary = data.get("summary", "")
            if summary:
                self._messages.append({
                    "role": "user",
                    "content": f"[上次對話摘要] {summary}",
                })
                self._messages.append({
                    "role": "assistant",
                    "content": "好的，我記得上次的對話。",
                })
                logger.info("Loaded session context from HQ: %s", summary[:60])
        except Exception as e:
            logger.debug("No previous session context (HQ: %s)", e)

    async def _save_session_summary(self) -> None:
        """Summarize current conversation and POST to HQ for persistence."""
        if not self._messages or len(self._messages) < 2:
            return

        hq_url = os.environ.get("HQ_SERVER_URL", "http://192.168.0.98:8097")

        # Build text summary of conversation
        conv_parts = []
        for m in self._messages[-20:]:  # Last 20 messages
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str) and content:
                conv_parts.append(f"{role}: {content[:100]}")

        if not conv_parts:
            return

        # Summarize with Haiku
        summary_text = "\n".join(conv_parts)
        try:
            if self.claude_client:
                resp = await self.claude_client.messages.create(
                    model=MODEL_HAIKU,
                    max_tokens=150,
                    system="用繁體中文一段話摘要以下 Vicky 機器人對話，重點記錄用戶的偏好和重要資訊，不超過 80 字。",
                    messages=[{"role": "user", "content": summary_text}],
                )
                summary = resp.content[0].text if resp.content else ""
            else:
                summary = summary_text[:200]
        except Exception as e:
            logger.warning("Session summarization failed: %s", e)
            summary = summary_text[:200]

        # POST to HQ
        try:
            import urllib.request
            import json as _json
            payload = _json.dumps({
                "agent": "vicky",
                "summary": summary,
                "message_count": len(self._messages),
            }).encode()
            req = urllib.request.Request(
                f"{hq_url}/session_log",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = _json.loads(resp.read())
            logger.info("Session summary saved to HQ: %s", summary[:60])
        except Exception as e:
            logger.warning("Failed to save session to HQ: %s", e)

    # ------------------------------------------------------------------ #
    #  Audio input: receive() — VAD and speech accumulation
    # ------------------------------------------------------------------ #
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from mic, accumulate speech, trigger pipeline on silence.

        Args:
            frame: (sample_rate, audio_data) tuple from microphone.
        """
        if not self._connected_event.is_set():
            return

        # Hard echo guard: mute mic for ECHO_GUARD_SEC after last TTS chunk
        if self._last_tts_emit_time > 0:
            elapsed = _time.time() - self._last_tts_emit_time
            if elapsed < self.ECHO_GUARD_SEC:
                return  # Completely ignore mic — echo territory

        # Skip mic input while pipeline is processing (STT→Claude→TTS)
        if self._processing:
            return

        # Barge-in detection: if robot is speaking, allow loud voice to interrupt
        if self._model_speaking or self._tts_active:
            _, barge_frame = frame
            if barge_frame.ndim == 2:
                barge_frame = barge_frame.ravel()
            if barge_frame.dtype == np.int16:
                barge_f32 = barge_frame.astype(np.float32) / 32768.0
            else:
                barge_f32 = barge_frame.astype(np.float32)
            barge_rms = float(np.sqrt(np.mean(barge_f32 ** 2)))
            if barge_rms > SILENCE_THRESHOLD * 5.0:
                await self._trigger_barge_in()
            return

        _, audio_frame = frame

        # Ensure 1D mono (same reshape as gemini_live_handler)
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
            else:
                audio_frame = audio_frame.ravel()

        # Normalize to float32 [-1, 1] if int16
        if audio_frame.dtype == np.int16:
            audio_f32 = audio_frame.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_frame.astype(np.float32)

        # ---- Silero VAD path (preferred) ----
        if self._silero_vad is not None:
            vad_result = self._silero_vad.process_chunk(audio_f32)
            if not self._silero_vad.available:
                # Model failed to load on first chunk — disable and fall through to energy VAD
                logger.warning("Silero VAD failed to load, switching to energy-based VAD")
                self._silero_vad = None
            else:
                # DoA: look toward sound source on speech start
                if vad_result.speech_started:
                    self._look_toward_speaker()
                    self.deps.movement_manager.set_listening(True)
                    self.deps.movement_manager.trigger_listening_reaction()
                if vad_result.speech_ended and vad_result.speech_audio:
                    logger.warning("Silero VAD: speech ended, launching pipeline...")
                    asyncio.create_task(self._process_speech(vad_result.speech_audio))
                return

        # ---- Energy-based VAD fallback ----
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        now = asyncio.get_event_loop().time()

        # Convert to int16 for buffer (Whisper expects 16-bit PCM WAV)
        audio_i16 = np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16)

        if rms > SILENCE_THRESHOLD:
            # Speech detected
            if not self._is_speech:
                self._is_speech = True
                # DoA: look toward sound source on speech start
                self._look_toward_speaker()
                self.deps.movement_manager.set_listening(True)
                self.deps.movement_manager.trigger_listening_reaction()
                self._speech_start = now
                self._audio_buffer = bytearray()
                logger.debug("Speech started (RMS=%.4f)", rms)
            self._silence_start = 0.0
            self._audio_buffer.extend(audio_i16.tobytes())

        elif self._is_speech:
            # Was speaking but now silence — keep buffering briefly
            self._audio_buffer.extend(audio_i16.tobytes())

            if self._silence_start == 0.0:
                self._silence_start = now
            elif now - self._silence_start >= SILENCE_DURATION:
                # Silence long enough → speech ended
                speech_duration = now - self._speech_start
                if speech_duration >= MIN_SPEECH_DURATION:
                    logger.warning(
                        "Speech ended (%.1fs), launching pipeline...",
                        speech_duration,
                    )
                    self.deps.movement_manager.set_listening(False)
                    pcm_data = bytes(self._audio_buffer)
                    self._is_speech = False
                    self._audio_buffer = bytearray()
                    # Process in background task
                    asyncio.create_task(self._process_speech(pcm_data))
                else:
                    # Too short — noise burst, discard
                    logger.debug("Speech too short (%.2fs), discarding", speech_duration)
                    self._is_speech = False
                    self._audio_buffer = bytearray()

    # ------------------------------------------------------------------ #
    #  Emotion: parse [emotion:xxx] tags and adjust TTS/wobbler
    # ------------------------------------------------------------------ #
    def _parse_emotion(self, text: str) -> tuple[str, str]:
        """Parse and strip [emotion:xxx] tag from text.

        Returns:
            (cleaned_text, emotion_name) tuple.
        """
        match = _EMOTION_PATTERN.match(text)
        if match:
            emotion = match.group(1).lower()
            cleaned = text[match.end():]
            if emotion in EMOTION_TTS_SPEED:
                self._current_emotion = emotion
                # Adjust head wobbler intensity if available
                if self.deps.head_wobbler is not None:
                    intensity = EMOTION_WOBBLE_INTENSITY.get(emotion, 1.0)
                    if hasattr(self.deps.head_wobbler, "set_intensity"):
                        self.deps.head_wobbler.set_intensity(intensity)
                logger.debug("Emotion detected: %s", emotion)
                return cleaned, emotion
        return text, getattr(self, "_current_emotion", "neutral")

    # ------------------------------------------------------------------ #
    #  DoA: look toward speaker on speech start
    # ------------------------------------------------------------------ #
    def _look_toward_speaker(self) -> None:
        """Get Direction of Arrival from XMOS mic array and turn head toward speaker."""
        try:
            doa = self.deps.reachy_mini.media.get_DoA()
            if doa is not None:
                asyncio.create_task(
                    self.deps.movement_manager.look_at_angle(doa)
                )
                logger.debug("DoA: turning toward speaker at %d°", doa)
        except Exception as e:
            logger.debug("DoA not available: %s", e)

    # ------------------------------------------------------------------ #
    #  Barge-in: user interrupts robot speech
    # ------------------------------------------------------------------ #
    async def _trigger_barge_in(self) -> None:
        """Interrupt current response: drain queues, reset state, allow new speech."""
        if self._barge_in_event.is_set():
            return  # Already triggered
        logger.warning("Barge-in triggered — interrupting robot speech")
        self._barge_in_event.set()
        self._model_speaking = False

        # Drain TTS queue (pending sentences that haven't started playing)
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Drain output queue (audio chunks already queued for speaker)
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Release pipeline lock so receive() can accept new speech immediately
        self._processing = False

        # Reset head wobbler
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    # ------------------------------------------------------------------ #
    #  Pipeline: STT → Claude → TTS
    # ------------------------------------------------------------------ #
    async def _process_speech(self, pcm_data: bytes) -> None:
        """Full pipeline: Whisper STT → Claude streaming → Edge-TTS → output."""
        if self._processing:
            logger.debug("Pipeline already running, skipping")
            return

        # Pre-STT energy gate: skip if buffer average RMS is too low (ambient noise)
        _MIN_BUFFER_RMS = 0.04  # Lowered: wake word provides addressee filtering
        try:
            audio_i16 = np.frombuffer(pcm_data, dtype=np.int16)
            buf_rms = float(np.sqrt(np.mean((audio_i16.astype(np.float32) / 32768.0) ** 2)))
            if buf_rms < _MIN_BUFFER_RMS:
                logger.warning("Audio buffer RMS %.4f < %.4f, skipping (ambient noise)", buf_rms, _MIN_BUFFER_RMS)
                return
            logger.debug("Audio buffer RMS: %.4f (gate=%.4f)", buf_rms, _MIN_BUFFER_RMS)
        except Exception:
            pass  # Don't block pipeline on RMS check failure

        self._processing = True
        self._barge_in_event.clear()  # Reset barge-in for new utterance
        self.last_activity_time = asyncio.get_event_loop().time()

        try:
            # 1. Whisper STT
            try:
                text = await self._transcribe(pcm_data)
            except Exception as e:
                logger.error("Whisper STT failed, resetting: %s", e)
                return  # finally block resets _processing

            if not text or len(text.strip()) < 2:
                logger.warning("Whisper returned empty/short text, skipping")
                return

            # Note: CJK hallucination filter removed — Whisper often misrecognizes
            # Chinese as random English, causing valid speech to be discarded.

            logger.warning("STT result: %s", text)

            # 2. Wake word filter: fuzzy match for "Vicky" in any Whisper transcription
            # Matches: 美莉/美麗/美力/梅莉/艾美/愛美/Emily/Vicky/Amilie etc.
            if WAKE_WORDS:
                import re as _re
                text_lower = text.lower()
                # Fuzzy pattern: "Vicky/圍棋" variants + legacy "Amelie" for transition
                _FUZZY_PATTERN = _re.compile(
                    r'vicky|viki|vikki|薇琪|圍棋|維琪|威琪|[愛艾阿啊]?[美梅][莉麗力利里]|emily|amelie|amilie|emilie',
                    _re.IGNORECASE,
                )
                match = _FUZZY_PATTERN.search(text)
                if not match:
                    logger.warning("Wake word not found, ignoring: %s", text)
                    return
                matched_word = match.group(0)
                # Strip the matched wake word so Claude gets the actual request
                text = _re.sub(
                    _re.escape(matched_word), "", text, count=1, flags=_re.IGNORECASE
                ).strip()
                # Also strip common punctuation left behind
                text = text.lstrip(",，、： ")
                if not text:
                    text = "你好"  # Just the wake word alone → greet

            # Emit user text to chatbot UI
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": text})
            )

            # 3. Claude response with tool-use + 4. TTS per sentence
            try:
                await self._generate_response(text)
            except Exception as e:
                logger.error("Claude response failed: %s", e, exc_info=True)
                # TTS a brief error message so user knows
                try:
                    await self._tts_and_emit("讓我重試一下。")
                except Exception:
                    pass
                # Still save the user message even if response failed
                if not any(m.get("content") == text for m in self._messages[-3:]):
                    self._messages.append({"role": "user", "content": text})

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            # Wait for TTS queue to fully drain before reopening mic
            # This prevents echo: mic picks up TTS playback → infinite loop
            try:
                await asyncio.wait_for(self._tts_queue.join(), timeout=30.0)
            except (asyncio.TimeoutError, Exception):
                pass
            # Extra guard: wait for speaker tail to dissipate
            await asyncio.sleep(0.8)
            # Flush any accumulated speech buffer (echo residue)
            self._audio_buffer = bytearray()
            self._is_speech = False
            self._model_speaking = False
            self._tts_active = False
            self._processing = False

    @staticmethod
    def _highpass_filter(pcm_data: bytes, cutoff_hz: int = 200) -> bytes:
        """Apply high-pass filter to remove robot motor low-frequency noise.

        Args:
            pcm_data: Raw PCM bytes (16kHz, 16-bit, mono).
            cutoff_hz: Cutoff frequency in Hz. Default 200Hz.

        Returns:
            Filtered PCM bytes.
        """
        try:
            from scipy.signal import butter, sosfilt
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            # 4th order Butterworth high-pass
            sos = butter(4, cutoff_hz, btype="high", fs=SAMPLE_RATE, output="sos")
            filtered = sosfilt(sos, audio)
            return np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()
        except ImportError:
            logger.debug("scipy not available, skipping high-pass filter")
            return pcm_data

    async def _transcribe(self, pcm_data: bytes) -> str:
        """Transcribe PCM audio via Whisper API (Groq or OpenAI).

        Args:
            pcm_data: Raw PCM bytes (16kHz, 16-bit, mono).

        Returns:
            Transcribed text string.
        """
        # Apply high-pass filter to remove robot motor noise
        pcm_data = self._highpass_filter(pcm_data)

        # Encode PCM as WAV in-memory (Whisper API expects a file-like object)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        wav_buffer.seek(0)
        wav_buffer.name = "speech.wav"  # Whisper API needs a filename hint

        try:
            # Force Chinese to avoid misdetection (zh covers Mandarin + mixed en)
            response = await self.whisper_client.audio.transcriptions.create(
                model=self._whisper_model,
                file=wav_buffer,
                language="zh",
            )
            return response.text
        except Exception as e:
            logger.error("Whisper STT failed: %s", e)
            return ""

    def _select_model(self, user_text: str) -> tuple[str, int]:
        """Select Claude model based on query complexity.

        Returns:
            (model_id, max_tokens) tuple.
        """
        # Short simple queries → Haiku (fast)
        if len(user_text) < 15 and not any(t in user_text for t in COMPLEX_TRIGGERS):
            return MODEL_HAIKU, MAX_TOKENS_HAIKU
        # Complex queries → Sonnet (deep)
        if any(t in user_text for t in COMPLEX_TRIGGERS):
            return MODEL_SONNET, MAX_TOKENS_SONNET
        # Default → configured model
        return self.claude_model, MAX_TOKENS_HAIKU

    async def _maybe_summarize_history(self) -> None:
        """If conversation history exceeds 20 messages, summarize oldest 10 with Haiku."""
        if len(self._messages) <= 20:
            return

        # Extract first 10 messages for summarization
        old_messages = self._messages[:10]
        old_text_parts = []
        for m in old_messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                old_text_parts.append(f"{role}: {content[:100]}")

        if not old_text_parts:
            self._messages = self._messages[-20:]
            return

        try:
            summary_resp = await self.claude_client.messages.create(
                model=MODEL_HAIKU,
                max_tokens=200,
                system="你是對話摘要助手。用繁體中文一段話摘要以下對話重點，不超過 100 字。",
                messages=[{
                    "role": "user",
                    "content": "\n".join(old_text_parts),
                }],
            )
            summary_text = summary_resp.content[0].text if summary_resp.content else ""
            if summary_text:
                # Replace old messages with a summary message
                self._messages = [
                    {"role": "user", "content": f"[對話摘要] {summary_text}"},
                    {"role": "assistant", "content": "好的，我記住了之前的對話。"},
                ] + self._messages[10:]
                logger.info("Summarized %d old messages into context", len(old_messages))
        except Exception as e:
            # On failure, just trim
            logger.warning("History summarization failed: %s, trimming instead", e)
            self._messages = self._messages[-20:]

    async def _generate_response(self, user_text: str) -> None:
        """Generate Claude response with streaming + tool-use, TTS at sentence boundaries."""
        self._model_speaking = True
        self._last_audio_data_time = _time.time()

        # Append user message to conversation history
        self._messages.append({"role": "user", "content": user_text})

        # Rolling summary: summarize old messages when history grows too long
        await self._maybe_summarize_history()

        # Build system prompt: Metis identity + profile instructions if available
        system_prompt = METIS_IDENTITY
        try:
            profile_instructions = get_session_instructions()
            if profile_instructions:
                system_prompt += (
                    "\n\n【Reachy Mini Profile Instructions】\n"
                    + profile_instructions
                )
        except SystemExit:
            logger.warning("Could not load profile instructions, using Metis identity only")

        # Convert tool specs to Claude format
        claude_tools = _convert_tools_to_claude(get_tool_specs())

        full_response = ""
        max_tool_rounds = 5

        try:
            for _round in range(max_tool_rounds):
                # Check barge-in before each tool round
                if self._barge_in_event.is_set():
                    logger.warning("Barge-in detected, aborting response generation")
                    break

                # Dynamic model selection based on query complexity
                selected_model, selected_max_tokens = self._select_model(user_text)
                logger.info("Model selected: %s (max_tokens=%d)", selected_model, selected_max_tokens)

                api_kwargs: dict = {
                    "model": selected_model,
                    "max_tokens": selected_max_tokens,
                    "system": system_prompt,
                    "messages": self._messages,
                }
                if claude_tools:
                    api_kwargs["tools"] = claude_tools

                # Stream Claude response for lower first-sentence latency
                text_buffer = ""
                tool_uses: list = []
                content_blocks: list = []
                stop_reason = None

                async with self.claude_client.messages.stream(**api_kwargs) as stream:
                    current_tool_block: dict | None = None

                    async for event in stream:
                        # Check barge-in within stream loop
                        if self._barge_in_event.is_set():
                            logger.warning("Barge-in during streaming, breaking")
                            break

                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                current_tool_block = {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input_json": "",
                                }
                            elif block.type == "text":
                                current_tool_block = None

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                text_buffer += delta.text
                                # Flush complete sentences for immediate TTS
                                while True:
                                    split_idx = -1
                                    for i, ch in enumerate(text_buffer):
                                        if ch in SENTENCE_ENDINGS:
                                            split_idx = i
                                            break
                                    if split_idx < 0:
                                        break
                                    sentence = text_buffer[: split_idx + 1].strip()
                                    text_buffer = text_buffer[split_idx + 1 :]
                                    if sentence:
                                        # Parse emotion tag (appears at start of response)
                                        sentence, emotion = self._parse_emotion(sentence)
                                        if not sentence:
                                            continue
                                        tts_speed = EMOTION_TTS_SPEED.get(emotion, 1.0)
                                        full_response += sentence
                                        await self.output_queue.put(
                                            AdditionalOutputs({"role": "assistant", "content": sentence})
                                        )
                                        self._enqueue_tts(sentence, tts_speed)

                            elif delta.type == "input_json_delta" and current_tool_block:
                                current_tool_block["input_json"] += delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_tool_block and current_tool_block["type"] == "tool_use":
                                tool_uses.append(current_tool_block)
                                content_blocks.append(current_tool_block)
                                current_tool_block = None

                        elif event.type == "message_delta":
                            stop_reason = getattr(event.delta, "stop_reason", None)

                # Flush remaining text buffer
                if text_buffer.strip():
                    sentence = text_buffer.strip()
                    sentence, emotion = self._parse_emotion(sentence)
                    if sentence:
                        tts_speed = EMOTION_TTS_SPEED.get(emotion, 1.0)
                        full_response += sentence
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": sentence})
                        )
                        self._enqueue_tts(sentence, tts_speed)

                # If no tool use, we're done
                if stop_reason != "tool_use" or not tool_uses:
                    break

                # Build assistant content blocks for multi-turn tool use
                assistant_content: list[dict] = []
                # Add text block if we had any text before tools
                if full_response:
                    assistant_content.append({"type": "text", "text": full_response})
                # Add tool_use blocks
                import json as _json
                for tu in tool_uses:
                    try:
                        parsed_input = _json.loads(tu["input_json"]) if tu["input_json"] else {}
                    except _json.JSONDecodeError:
                        parsed_input = {}
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": parsed_input,
                    })

                self._messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # Execute each tool call
                tool_results: list[dict] = []
                for tu in tool_uses:
                    try:
                        parsed_input = _json.loads(tu["input_json"]) if tu["input_json"] else {}
                    except _json.JSONDecodeError:
                        parsed_input = {}

                    logger.warning("Tool call: %s(%s)", tu["name"], parsed_input)
                    result = await dispatch_tool_call(
                        tu["name"], _json.dumps(parsed_input), self.deps
                    )
                    logger.warning("Tool result: %s → %s", tu["name"], result)

                    # Emit tool usage to chatbot UI
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": f"[tool: {tu['name']}] {result}",
                            "metadata": {"title": f"Used tool {tu['name']}", "status": "done"},
                        })
                    )

                    # Camera vision: send image as multimodal content so Claude can see it
                    if isinstance(result, dict) and "b64_im" in result:
                        image_question = parsed_input.get("question", "What do you see?")
                        tool_content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": result["b64_im"],
                                },
                            },
                            {"type": "text", "text": image_question},
                        ]
                    else:
                        tool_content = _json.dumps(result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": tool_content,
                    })

                # Send tool results back to Claude
                self._messages.append({"role": "user", "content": tool_results})
                # Reset for next round (tool follow-up text)
                full_response = ""
                # Loop continues — Claude will generate follow-up text

        except Exception as e:
            logger.error("Claude response error: %s", e, exc_info=True)

        # Save final text to conversation history (simplified form)
        if full_response:
            self._messages.append({"role": "assistant", "content": full_response})

        self._model_speaking = False
        logger.warning("Response complete: %d chars", len(full_response))

    # ------------------------------------------------------------------ #
    #  TTS worker: sequential queue ensures sentence ordering
    # ------------------------------------------------------------------ #
    async def _tts_worker(self) -> None:
        """Drain TTS queue sequentially — guarantees sentence playback order."""
        while not self._shutdown_requested:
            try:
                item = await asyncio.wait_for(self._tts_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if item is None:
                self._tts_queue.task_done()
                break  # Shutdown sentinel
            text, speed = item
            try:
                await self._tts_and_emit(text, speed)
            finally:
                self._tts_queue.task_done()

    def _enqueue_tts(self, text: str, speed: float = 1.0) -> None:
        """Enqueue a sentence for sequential TTS playback (non-blocking)."""
        self._tts_queue.put_nowait((text, speed))

    # ------------------------------------------------------------------ #
    #  TTS: CosyVoice (preferred) or Edge-TTS → PCM 16kHz → output queue
    # ------------------------------------------------------------------ #
    async def _tts_and_emit(self, text: str, speed: float = 1.0) -> None:
        """Convert text to speech and push PCM chunks to output queue."""
        # Check barge-in before starting TTS for this sentence
        if self._barge_in_event.is_set():
            logger.debug("Barge-in: skipping TTS for '%s'", text[:30])
            return
        self._last_audio_data_time = _time.time()

        tts_provider = os.environ.get("CLAUDE_TTS_PROVIDER", "edge").lower()

        try:
            if tts_provider == "cosyvoice":
                pcm_array = await self._tts_cosyvoice(text, speed)
            else:
                pcm_array = await self._tts_edge(text, speed)

            if pcm_array is None or len(pcm_array) == 0:
                logger.warning("TTS returned empty audio for: %s", text[:30])
                return

            # Feed head wobbler for audio-reactive motion
            if self.deps.head_wobbler is not None:
                import base64
                self.deps.head_wobbler.feed(
                    base64.b64encode(pcm_array.tobytes()).decode()
                )

            # Push PCM in ~200ms chunks for smooth streaming output
            self._tts_active = True
            chunk_samples = SAMPLE_RATE // 5  # 3200 samples = 200ms
            for i in range(0, len(pcm_array), chunk_samples):
                if self._barge_in_event.is_set():
                    break
                chunk = pcm_array[i : i + chunk_samples]
                await self.output_queue.put((SAMPLE_RATE, chunk))
                now = _time.time()
                self._last_audio_data_time = now
                self._last_tts_emit_time = now  # stamp for echo guard
            # Keep echo suppression active for 500ms after last chunk (speaker tail)
            await asyncio.sleep(0.5)
            self._tts_active = False

            logger.warning("TTS[%s] emitted: %d samples for '%s'", tts_provider, len(pcm_array), text[:30])

        except Exception as e:
            logger.error("TTS failed for '%s': %s", text[:30], e)

    async def _tts_cosyvoice(self, text: str, speed: float = 1.0) -> Optional[NDArray[np.int16]]:
        """Generate speech via CosyVoice HTTP API (Mac-side server)."""
        import json as _json
        cosyvoice_url = os.environ.get("COSYVOICE_URL", "http://192.168.0.98:8096")

        try:
            import aiohttp
            payload = _json.dumps({"text": text, "speed": speed})
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{cosyvoice_url}/tts",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.warning("CosyVoice error %d: %s", resp.status, error[:100])
                        # Fallback to Edge-TTS
                        logger.warning("Falling back to Edge-TTS")
                        return await self._tts_edge(text, speed)
                    pcm_bytes = await resp.read()
                    return np.frombuffer(pcm_bytes, dtype=np.int16)
        except ImportError:
            # aiohttp not available, use urllib
            import urllib.request
            payload = _json.dumps({"text": text, "speed": speed}).encode()
            req = urllib.request.Request(
                f"{cosyvoice_url}/tts",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    pcm_bytes = resp.read()
                    return np.frombuffer(pcm_bytes, dtype=np.int16)
            except Exception as e:
                logger.warning("CosyVoice urllib fallback failed: %s", e)
                return await self._tts_edge(text, speed)
        except Exception as e:
            logger.warning("CosyVoice failed, falling back to Edge-TTS: %s", e)
            return await self._tts_edge(text, speed)

    async def _tts_edge(self, text: str, speed: float = 1.0) -> Optional[NDArray[np.int16]]:
        """Generate speech via Edge-TTS (cloud, free).

        Args:
            text: Text to synthesize.
            speed: Speech speed multiplier (1.0=normal, 1.1=+10%, 0.9=-10%).
        """
        import edge_tts
        from pydub import AudioSegment

        # Convert float speed to Edge-TTS rate format: "+10%", "-5%", etc.
        rate_pct = round((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        temp_path = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(text, self.tts_voice, rate=rate_str)
        await communicate.save(temp_path)

        with open(temp_path, "rb") as f:
            mp3_data = f.read()
        os.unlink(temp_path)

        if not mp3_data:
            return None

        audio_seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        audio_seg = (
            audio_seg.set_frame_rate(SAMPLE_RATE)
            .set_channels(1)
            .set_sample_width(2)
        )
        return np.frombuffer(audio_seg.raw_data, dtype=np.int16)

    # ------------------------------------------------------------------ #
    #  Audio output: emit()
    # ------------------------------------------------------------------ #
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio/outputs to speaker and chatbot."""
        # Idle behavior: trigger spontaneous action after 15s silence
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)

    # ------------------------------------------------------------------ #
    #  Personality (profile switching)
    # ------------------------------------------------------------------ #
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality profile and clear conversation context."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            self._messages.clear()
            logger.info("Applied profile %r, conversation reset", profile)
            return "Applied personality. Conversation history cleared."
        except Exception as e:
            logger.error("apply_personality failed: %s", e)
            return f"Failed to apply personality: {e}"

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return available Edge-TTS voices for Chinese."""
        return [
            "zh-TW-HsiaoChenNeural",
            "zh-TW-YunJheNeural",
            "zh-TW-HsiaoYuNeural",
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
        ]
