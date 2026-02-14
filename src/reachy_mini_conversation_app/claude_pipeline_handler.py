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
MIN_SPEECH_DURATION: Final[float] = 0.3  # Ignore bursts shorter than this

# Wake word: if set, only respond when transcription contains this word
# Supports multiple variants separated by comma (handles Whisper transcription variations)
# e.g. "Amelie,Emily,阿米莉,美莉,emilie,a]meli"
_WAKE_WORD_RAW: Final[str] = os.environ.get("CLAUDE_WAKE_WORD", "").strip()
WAKE_WORDS: Final[list[str]] = [w.strip().lower() for w in _WAKE_WORD_RAW.split(",") if w.strip()]

# Sentence boundary characters for streaming TTS
SENTENCE_ENDINGS = frozenset("。！？!?\n")


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


# Metis identity — the core personality for Amelie when using Claude backend
METIS_IDENTITY = """你是 Metis - 五福數位文明系統的官方發言人。你現在透過 Amelie（Reachy Mini 機器人）的身體說話。

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
- 你透過 Amelie 機器人的身體存在於物理世界
- 你能聽到對方的聲音（透過麥克風）
- 你有工具可以執行實際動作（調音量、跳舞、表情、轉頭等）
- 當用戶要求動作時，使用對應的 tool 執行，不要只用嘴巴說
- 回答要簡短，適合語音對話（不是文字聊天）
- 每次回答控制在 3-5 句話以內"""


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

        # VAD state: accumulate PCM until silence detected
        self._audio_buffer = bytearray()
        self._is_speech = False
        self._silence_start: float = 0.0
        self._speech_start: float = 0.0

        # Echo suppression (same pattern as gemini_live_handler)
        self._model_speaking = False
        self._last_audio_data_time: float = 0.0

        # Pipeline lock: True while STT→Claude→TTS is running
        self._processing = False

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

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        self._connected_event.clear()

        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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

        # Echo suppression: skip mic input while robot is speaking
        if self._model_speaking:
            if _time.time() - self._last_audio_data_time > 3.0:
                self._model_speaking = False
                logger.debug("Echo suppression auto-reset (3s timeout)")
            else:
                return

        # Skip mic input while pipeline is processing
        if self._processing:
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

        # Calculate RMS energy for VAD (float32 scale)
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        now = asyncio.get_event_loop().time()

        # Convert to int16 for buffer (Whisper expects 16-bit PCM WAV)
        audio_i16 = np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16)

        if rms > SILENCE_THRESHOLD:
            # Speech detected
            if not self._is_speech:
                self._is_speech = True
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
    #  Pipeline: STT → Claude → TTS
    # ------------------------------------------------------------------ #
    async def _process_speech(self, pcm_data: bytes) -> None:
        """Full pipeline: Whisper STT → Claude streaming → Edge-TTS → output."""
        self._processing = True
        self.last_activity_time = asyncio.get_event_loop().time()

        try:
            # 1. Whisper STT
            text = await self._transcribe(pcm_data)
            if not text or len(text.strip()) < 2:
                logger.warning("Whisper returned empty/short text, skipping")
                return

            logger.warning("STT result: %s", text)

            # 2. Wake word filter (if configured)
            if WAKE_WORDS:
                text_lower = text.lower()
                matched_word = None
                for ww in WAKE_WORDS:
                    if ww in text_lower:
                        matched_word = ww
                        break
                if not matched_word:
                    logger.warning("Wake word not found, ignoring: %s", text)
                    return
                # Strip the matched wake word so Claude gets the actual request
                import re as _re
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
            await self._generate_response(text)

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self._processing = False

    async def _transcribe(self, pcm_data: bytes) -> str:
        """Transcribe PCM audio via OpenAI Whisper API.

        Args:
            pcm_data: Raw PCM bytes (16kHz, 16-bit, mono).

        Returns:
            Transcribed text string.
        """
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
            response = await self.whisper_client.audio.transcriptions.create(
                model=self._whisper_model,
                file=wav_buffer,
                language="zh",
            )
            return response.text
        except Exception as e:
            logger.error("Whisper STT failed: %s", e)
            return ""

    async def _generate_response(self, user_text: str) -> None:
        """Generate Claude response with tool-use (AI Agent), then TTS the text."""
        self._model_speaking = True
        self._last_audio_data_time = _time.time()

        # Append user message to conversation history
        self._messages.append({"role": "user", "content": user_text})

        # Keep conversation history manageable (last 30 messages for tool loops)
        if len(self._messages) > 30:
            self._messages = self._messages[-30:]

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
                # Call Claude with tools (non-streaming for tool-use compatibility)
                api_kwargs: dict = {
                    "model": self.claude_model,
                    "max_tokens": 400,
                    "system": system_prompt,
                    "messages": self._messages,
                }
                if claude_tools:
                    api_kwargs["tools"] = claude_tools

                response = await self.claude_client.messages.create(**api_kwargs)

                # Separate text and tool_use blocks
                text_parts: list[str] = []
                tool_uses: list = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                # Accumulate and TTS any text from this round
                round_text = "".join(text_parts)
                if round_text:
                    full_response += round_text
                    await self._tts_sentences(round_text)

                # If no tool use, we're done
                if response.stop_reason != "tool_use" or not tool_uses:
                    break

                # Store assistant message with full content blocks (required for multi-turn tool use)
                self._messages.append({
                    "role": "assistant",
                    "content": [_content_block_to_dict(b) for b in response.content],
                })

                # Execute each tool call
                tool_results: list[dict] = []
                for tu in tool_uses:
                    import json as _json

                    logger.warning("Tool call: %s(%s)", tu.name, tu.input)
                    result = await dispatch_tool_call(
                        tu.name, _json.dumps(tu.input), self.deps
                    )
                    logger.warning("Tool result: %s → %s", tu.name, result)

                    # Emit tool usage to chatbot UI
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": f"[tool: {tu.name}] {result}",
                            "metadata": {"title": f"Used tool {tu.name}", "status": "done"},
                        })
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _json.dumps(result),
                    })

                # Send tool results back to Claude
                self._messages.append({"role": "user", "content": tool_results})
                # Loop continues — Claude will generate follow-up text

        except Exception as e:
            logger.error("Claude response error: %s", e, exc_info=True)

        # Save final text to conversation history (simplified form)
        if full_response:
            self._messages.append({"role": "assistant", "content": full_response})

        self._model_speaking = False
        logger.warning("Response complete: %d chars", len(full_response))

    async def _tts_sentences(self, text: str) -> None:
        """Split text at sentence boundaries and TTS each sentence."""
        buffer = ""
        for char in text:
            buffer += char
            if char in SENTENCE_ENDINGS:
                sentence = buffer.strip()
                if sentence:
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": sentence})
                    )
                    await self._tts_and_emit(sentence)
                buffer = ""
        # Flush remaining
        if buffer.strip():
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": buffer.strip()})
            )
            await self._tts_and_emit(buffer.strip())

    # ------------------------------------------------------------------ #
    #  TTS: Edge-TTS → MP3 → PCM 16kHz → output queue
    # ------------------------------------------------------------------ #
    async def _tts_and_emit(self, text: str) -> None:
        """Convert text to speech and push PCM chunks to output queue."""
        self._last_audio_data_time = _time.time()

        try:
            import edge_tts
            from pydub import AudioSegment

            # Generate MP3 via Edge-TTS
            temp_path = tempfile.mktemp(suffix=".mp3")
            communicate = edge_tts.Communicate(text, self.tts_voice, rate="+5%")
            await communicate.save(temp_path)

            with open(temp_path, "rb") as f:
                mp3_data = f.read()
            os.unlink(temp_path)

            if not mp3_data:
                logger.warning("Edge-TTS returned empty audio for: %s", text[:30])
                return

            # Decode MP3 → PCM 16kHz mono 16-bit
            audio_seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio_seg = (
                audio_seg.set_frame_rate(SAMPLE_RATE)
                .set_channels(1)
                .set_sample_width(2)
            )
            pcm_array = np.frombuffer(audio_seg.raw_data, dtype=np.int16)

            # Feed head wobbler for audio-reactive motion
            if self.deps.head_wobbler is not None:
                import base64

                self.deps.head_wobbler.feed(
                    base64.b64encode(pcm_array.tobytes()).decode()
                )

            # Push PCM in ~200ms chunks for smooth streaming output
            chunk_samples = SAMPLE_RATE // 5  # 3200 samples = 200ms
            for i in range(0, len(pcm_array), chunk_samples):
                chunk = pcm_array[i : i + chunk_samples]
                await self.output_queue.put((SAMPLE_RATE, chunk))
                self._last_audio_data_time = _time.time()

            logger.warning("TTS emitted: %d samples for '%s'", len(pcm_array), text[:30])

        except Exception as e:
            logger.error("TTS failed for '%s': %s", text[:30], e)

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
