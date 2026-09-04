"""Adapter registry: pick a provider by name or infer it from the model id."""

from __future__ import annotations

from collections.abc import Callable

from toolassay.adapters.base import AdapterSettings, Conversation, Effort, ModelAdapter
from toolassay.core import ConfigError

AdapterFactory = Callable[[AdapterSettings], ModelAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register(provider: str, factory: AdapterFactory) -> None:
    """Make a provider available to ``create_adapter`` (also used by tests to plug in fakes)."""
    _REGISTRY[provider] = factory


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)


def detect_provider(model: str) -> str | None:
    """Infer the provider from a model id, or None when it is not recognisable."""
    if model.startswith("claude"):
        return "anthropic"
    return None


def create_adapter(settings: AdapterSettings, provider: str | None = None) -> ModelAdapter:
    """Build the adapter for ``settings.model``.

    ``provider`` overrides detection; pass it for model ids the registry cannot recognise.
    """
    name = provider or detect_provider(settings.model)
    if name is None:
        raise ConfigError(
            f"cannot infer a provider from model id {settings.model!r}; "
            f"pass --provider (registered: {', '.join(registered_providers())})"
        )
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ConfigError(
            f"unknown provider {name!r} (registered: {', '.join(registered_providers())})"
        )
    return factory(settings)


def _register_builtin() -> None:
    from toolassay.adapters.anthropic import AnthropicAdapter

    register("anthropic", AnthropicAdapter.from_settings)


_register_builtin()

__all__ = [
    "AdapterFactory",
    "AdapterSettings",
    "Conversation",
    "Effort",
    "ModelAdapter",
    "create_adapter",
    "detect_provider",
    "register",
    "registered_providers",
]
