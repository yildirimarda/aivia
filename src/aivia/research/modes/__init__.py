"""Topic modes, their protocol, base class and registry.

A mode discovers topic candidates according to one strategy. Importing this
package exposes the protocol and registry; the built-in modes register
themselves from their own modules.
"""

from aivia.research.modes.base import (
    BaseMode,
    BaseTopicMode,
    Mode,
    ModeBase,
    ModeRegistry,
    TopicMode,
    TopicModeBase,
    TopicModeCandidate,
    TopicModeProtocol,
    TopicModeRegistration,
    TopicModeRegistry,
    TopicModeRequest,
    TopicModeResponse,
    TopicModeResult,
    mode_registry,
    register_mode,
    register_topic_mode,
    resolve_mode,
    resolve_topic_mode,
    topic_mode_registry,
)

__all__ = [
    "BaseMode",
    "BaseTopicMode",
    "Mode",
    "ModeBase",
    "ModeRegistry",
    "TopicMode",
    "TopicModeBase",
    "TopicModeCandidate",
    "TopicModeProtocol",
    "TopicModeRegistration",
    "TopicModeRegistry",
    "TopicModeRequest",
    "TopicModeResponse",
    "TopicModeResult",
    "mode_registry",
    "register_mode",
    "register_topic_mode",
    "resolve_mode",
    "resolve_topic_mode",
    "topic_mode_registry",
]
