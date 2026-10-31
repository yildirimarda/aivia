"""Speech to text provider adapters.

Importing this package makes every built-in transcription adapter available to
the provider registry. A local faster-whisper adapter costs nothing, while the
hosted adapter records per-minute cost through
:class:`~aivia.providers.stt.cost.STTCostCalculator`.
"""

from aivia.providers.stt.common import (
    audio_filename,
    load_audio,
    response_duration_seconds,
    segments_from_values,
    words_from_values,
)
from aivia.providers.stt.cost import (
    MINUTE_UNIT,
    STT_OPERATION,
    STTCost,
    STTCostCalculator,
    STTPrice,
    stt_cost_event,
)
from aivia.providers.stt.local import (
    DEFAULT_MODEL,
    FasterWhisper,
    FasterWhisperProvider,
    FasterWhisperSTT,
    LocalSTT,
    LocalSTTProvider,
    Transcriber,
    WhisperLocal,
    WhisperLocalSTT,
)
from aivia.providers.stt.openai import (
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_STT_BASE_URL,
    OpenAISpeechToText,
    OpenAISpeechToTextProvider,
    OpenAISTT,
    OpenAISTTProvider,
    OpenAITranscription,
    OpenAITranscriptionProvider,
    Whisper,
    WhisperAPI,
    WhisperAPIProvider,
    WhisperSTT,
    WhisperSTTProvider,
)

__all__ = [
    "DEFAULT_MODEL",
    "MINUTE_UNIT",
    "OPENAI_DEFAULT_BASE_URL",
    "OPENAI_STT_BASE_URL",
    "STT_OPERATION",
    "FasterWhisper",
    "FasterWhisperProvider",
    "FasterWhisperSTT",
    "LocalSTT",
    "LocalSTTProvider",
    "OpenAISTT",
    "OpenAISTTProvider",
    "OpenAISpeechToText",
    "OpenAISpeechToTextProvider",
    "OpenAITranscription",
    "OpenAITranscriptionProvider",
    "STTCost",
    "STTCostCalculator",
    "STTPrice",
    "Transcriber",
    "Whisper",
    "WhisperAPI",
    "WhisperAPIProvider",
    "WhisperLocal",
    "WhisperLocalSTT",
    "WhisperSTT",
    "WhisperSTTProvider",
    "audio_filename",
    "load_audio",
    "response_duration_seconds",
    "segments_from_values",
    "stt_cost_event",
    "words_from_values",
]
