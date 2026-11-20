
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gpt-5-nano"
_PLACEHOLDERS = {"", "replace-me", "changeme"}


def read_env_file(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _credential(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in _PLACEHOLDERS or cleaned.startswith("replace"):
        return None
    return cleaned or None


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None = field(default=None, repr=False)
    llm_model: str = DEFAULT_MODEL
    llm_provider: str = "openai"
    llm_config_note: str | None = None
    embedding_provider: str = "specter2"
    embedding_model: str | None = None
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    db_path: Path = field(default_factory=lambda: Path("./data/litreview.sqlite3"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))
    contact_email: str | None = None
    http_user_agent: str = "litreview/0.1 (academic project)"
    log_level: str = "INFO"
    refusal_tau: float | None = None
    refusal_tau_mode: str = "dense"

    @classmethod
    def from_env(cls, env_path: str | Path | None = ".env") -> Settings:
        file_values = read_env_file(env_path) if env_path is not None else {}

        lowered = {name.lower(): value for name, value in file_values.items()}

        def get(key: str) -> str | None:
            if key in os.environ:
                return os.environ[key]
            env_ci = {name.lower(): value for name, value in os.environ.items()}
            if key.lower() in env_ci:
                return env_ci[key.lower()]
            return lowered.get(key.lower())

        raw_tau = get("REFUSAL_TAU")
        try:
            refusal_tau = float(raw_tau) if raw_tau not in (None, "") else None
        except ValueError:
            raise ValueError(f"REFUSAL_TAU must be a number, got {raw_tau!r}") from None
        tau_mode = get("REFUSAL_TAU_MODE") or "dense"
        if tau_mode not in {"lexical", "tfidf", "dense", "hybrid"}:
            raise ValueError(f"REFUSAL_TAU_MODE must be lexical|tfidf|dense|hybrid, got {tau_mode!r}")

        gemini_key = _credential(get("GEMINI_API_KEY")) or _credential(get("GOOGLE_API_KEY"))
        openai_key = _credential(get("OPENAI_API_KEY"))
        provider = (get("LLM_PROVIDER") or ("gemini" if gemini_key else "openai")).lower()
        if provider not in {"gemini", "openai"}:
            raise ValueError(f"LLM_PROVIDER must be gemini or openai, got {provider!r}")
        from services.llm import resolve_model

        model, note = resolve_model(provider, get("LLM_MODEL"))

        return cls(
            llm_api_key=gemini_key if provider == "gemini" else openai_key,
            llm_model=model,
            llm_provider=provider,
            llm_config_note=note,
            embedding_provider=get("EMBEDDING_PROVIDER") or "specter2",
            embedding_model=get("EMBEDDING_MODEL") or None,
            models_dir=Path(get("MODELS_DIR") or "./models"),
            db_path=Path(get("DB_PATH") or "./data/litreview.sqlite3"),
            data_dir=Path(get("DATA_DIR") or "./data"),
            cache_dir=Path(get("CACHE_DIR") or "./data/cache"),
            contact_email=_credential(get("CONTACT_EMAIL")),
            http_user_agent=get("HTTP_USER_AGENT") or "litreview/0.1 (academic project)",
            log_level=get("LOG_LEVEL") or "INFO",
            refusal_tau=refusal_tau,
            refusal_tau_mode=tau_mode,
        )

    @property
    def generation_available(self) -> bool:
        return self.llm_api_key is not None
