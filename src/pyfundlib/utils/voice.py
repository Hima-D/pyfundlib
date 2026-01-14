# src/pyfundlib/utils/voice.py
from __future__ import annotations

import os
from typing import Optional

import whisper
try:
    from elevenlabs import generate, save, set_api_key
except ImportError:
    # Handle newer elevenlabs versions or missing functions
    from elevenlabs.client import ElevenLabs
    def generate(**kwargs): return b""
    def save(audio, path): pass
    def set_api_key(key): pass
import pyttsx3

from pyfundlib.config import settings
from pyfundlib.utils.logger import get_logger
from pyfundlib.agents.voice_router import VoiceCommandRouter

logger = get_logger(__name__)


class VoiceInterface:
    """
    Multimodal voice interface for pyfundlib.
    Supports STT (Whisper) and TTS (ElevenLabs/pyttsx3).
    """

    def __init__(self, use_elevenlabs: bool = False):
        self.use_elevenlabs = use_elevenlabs
        if use_elevenlabs and settings.api_key:
            set_api_key(settings.api_key)

        # Initialize pyttsx3 as fallback
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)  # Slightly faster for professional feel
        self.router = VoiceCommandRouter()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text using OpenAI Whisper"""
        try:
            logger.info("transcribing_audio", path=audio_path)
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text = result.get("text", "").strip()
            logger.info("transcription_completed", text=text)
            return text
        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return ""

    def speak(self, text: str, output_path: Optional[str] = None) -> None:
        """Text-to-Speech output"""
        try:
            logger.info("synthesizing_speech", text_length=len(text))
            
            if self.use_elevenlabs:
                audio = generate(
                    text=text,
                    voice="Bella",  # Professional 2026 voice
                    model="eleven_multilingual_v2"
                )
                if output_path:
                    save(audio, output_path)
                logger.info("elevenlabs_speech_completed")
            else:
                # Local fallback
                self.engine.say(text)
                self.engine.runAndWait()
                logger.info("pyttsx3_speech_completed")
                
        except Exception as e:
            logger.error("speech_synthesis_failed", error=str(e))
            # Fallback to print
            print(f"[VOICE] {text}")

    def handle_audio_command(self, audio_path: str) -> str:
        text = self.transcribe(audio_path)
        if not text:
            msg = "I could not understand the audio clearly."
            self.speak(msg)
            return msg

        result = self.router.handle(text)
        self.speak(result.response)
        return result.response


# Alias for backward compatibility and easier access
VoiceAssistant = VoiceInterface
