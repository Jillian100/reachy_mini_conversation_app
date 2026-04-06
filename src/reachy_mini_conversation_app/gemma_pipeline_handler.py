"""Gemma 4 Local Pipeline handler for Reachy Mini conversations.

Pipeline: Mic (16kHz PCM) → VAD → Whisper STT → Gemma 4 (Ollama) → Edge-TTS → Speaker

Inherits all audio infrastructure (VAD, STT, TTS, barge-in, emotion, echo suppression)
from ClaudePipelineHandler. Only the LLM call path is replaced with local Ollama API.

Benefits:
- Zero LLM API cost (runs on M1 Max via Ollama)
- Low latency (no round-trip to cloud)
- Multimodal capable (Gemma 4 supports image input)
- Apache 2.0 licensed

Requires: openai (for Ollama OpenAI-compatible API), Ollama running with gemma4 model
"""

from __future__ import annotations

import os
import asyncio
import logging
import time as _time
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
from fastrtc import AdditionalOutputs
from numpy.typing import NDArray

from reachy_mini_conversation_app.claude_pipeline_handler import (
    ClaudePipelineHandler,
    METIS_IDENTITY,
    SENTENCE_ENDINGS,
    EMOTION_TTS_SPEED,
)
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# Ollama defaults
OLLAMA_BASE_URL: Final[str] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: Final[str] = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_MAX_TOKENS: Final[int] = int(os.environ.get("OLLAMA_MAX_TOKENS", "150"))


class GemmaPipelineHandler(ClaudePipelineHandler):
    """Gemma 4 local pipeline: Whisper STT → Ollama Gemma 4 → Edge-TTS.

    Inherits VAD, STT, TTS, barge-in, echo suppression from ClaudePipelineHandler.
    Only the LLM call is routed to local Ollama instead of Claude API.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        super().__init__(deps, gradio_mode, instance_path)
        self._ollama_client: Any = None

    def copy(self) -> GemmaPipelineHandler:
        return GemmaPipelineHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Initialize Ollama client + Whisper STT + TTS (reuse parent's STT/TTS setup)."""
        # --- Whisper STT (same as Claude pipeline) ---
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        from openai import AsyncOpenAI

        # STT client (Groq preferred, OpenAI fallback)
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
            logger.error("No GROQ_API_KEY or OPENAI_API_KEY for Whisper STT.")
            return

        # --- Ollama LLM client (OpenAI-compatible API) ---
        self._ollama_client = AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't check this, but the client requires it
            base_url=OLLAMA_BASE_URL,
        )

        # Verify Ollama is reachable
        try:
            models = await self._ollama_client.models.list()
            model_names = [m.id for m in models.data]
            if OLLAMA_MODEL not in model_names:
                logger.warning(
                    "Model %s not found in Ollama. Available: %s",
                    OLLAMA_MODEL, model_names,
                )
            else:
                logger.info("Ollama model verified: %s", OLLAMA_MODEL)
        except Exception as e:
            logger.error("Cannot reach Ollama at %s: %s", OLLAMA_BASE_URL, e)
            return

        # TTS voice
        self.tts_voice = os.environ.get("METIS_VOICE", "zh-TW-HsiaoChenNeural")

        # Claude client is still needed for session summary (optional)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if anthropic_key:
            try:
                import anthropic
                self.claude_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            except ImportError:
                pass

        loop = asyncio.get_event_loop()
        self.start_time = loop.time()
        self.last_activity_time = loop.time()
        self._connected_event.set()

        logger.warning(
            "Gemma pipeline ready: model=%s base_url=%s voice=%s stt=%s",
            OLLAMA_MODEL, OLLAMA_BASE_URL, self.tts_voice, self._whisper_model,
        )

        # Start TTS worker
        self._tts_worker_task = asyncio.create_task(self._tts_worker())

        # Load session context
        await self._load_session_context()

    async def _generate_response(self, user_text: str) -> None:
        """Generate response via local Ollama Gemma 4, TTS at sentence boundaries."""
        self._model_speaking = True
        self._last_audio_data_time = _time.time()

        # Append user message
        self._messages.append({"role": "user", "content": user_text})

        # Trim history if too long (simple trim, no Claude summarization)
        if len(self._messages) > 20:
            # Keep system context + last 16 messages
            self._messages = self._messages[:2] + self._messages[-16:]

        # Build system prompt
        system_prompt = METIS_IDENTITY
        try:
            profile_instructions = get_session_instructions()
            if profile_instructions:
                system_prompt += (
                    "\n\n【Reachy Mini Profile Instructions】\n"
                    + profile_instructions
                )
        except SystemExit:
            pass

        # Build messages for OpenAI-compatible API
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in self._messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                api_messages.append({"role": role, "content": content})

        full_response = ""
        text_buffer = ""

        try:
            stream = await self._ollama_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=api_messages,
                max_tokens=OLLAMA_MAX_TOKENS,
                temperature=0.7,
                stream=True,
                extra_body={"think": False},  # Disable thinking for low latency
            )

            async for chunk in stream:
                if self._barge_in_event.is_set():
                    logger.warning("Barge-in during streaming, breaking")
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text_buffer += delta.content

                    # Flush complete sentences for immediate TTS
                    while True:
                        split_idx = -1
                        for i, ch in enumerate(text_buffer):
                            if ch in SENTENCE_ENDINGS:
                                split_idx = i
                                break
                        if split_idx < 0:
                            break
                        sentence = text_buffer[:split_idx + 1].strip()
                        text_buffer = text_buffer[split_idx + 1:]
                        if sentence:
                            sentence, emotion = self._parse_emotion(sentence)
                            if not sentence:
                                continue
                            tts_speed = EMOTION_TTS_SPEED.get(emotion, 1.0)
                            full_response += sentence
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": sentence})
                            )
                            self._enqueue_tts(sentence, tts_speed)

            # Flush remaining buffer
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

        except Exception as e:
            logger.error("Ollama response error: %s", e, exc_info=True)

        # Save to conversation history
        if full_response:
            self._messages.append({"role": "assistant", "content": full_response})

        self._model_speaking = False
        logger.warning("Gemma response complete: %d chars", len(full_response))
