"""
Central configuration loaded from environment variables via pydantic-settings.
All values have safe defaults; secrets must be supplied via .env.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repo — el .env se busca aquí aunque el cwd sea otro
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider selector ────────────────────────────────────────────────
    llm_provider: str = Field("ollama", description="'ollama' para modelo local, 'gemini' para Google API")

    # ── Google Gemini (LLM) ───────────────────────────────────────────────────
    gemini_api_key: str = Field("", description="Google Gemini API key (gratis en aistudio.google.com)")
    gemini_model: str = Field("gemini-2.0-flash", description="Modelo Gemini a usar")

    # ── Ollama (LLM local) ────────────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", description="URL base de Ollama")
    ollama_model: str = Field("gemma3:4b", description="Modelo Ollama a usar")

    # ── OpenAI (opcional, ya no requerido) ────────────────────────────────────
    openai_api_key: str = Field("", description="OpenAI API key (opcional, no requerido)")
    openai_model: str = Field("gpt-4o-mini", description="Modelo OpenAI (si se usa)")
    openai_embedding_model: str = Field("text-embedding-3-small", description="Embedding OpenAI (si se usa)")

    # ── Polymarket wallet ─────────────────────────────────────────────────────
    polymarket_wallet_address: str = Field(
        "", description="0x address — used for position queries, never for signing here"
    )

    # ── macOS Keychain ────────────────────────────────────────────────────────
    keychain_service: str = Field(
        "polymarket_pk", description="Keychain service name for the private key"
    )
    keychain_account: str = Field(
        "bot_user", description="Keychain account name for the private key"
    )

    # ── Polygon RPC ───────────────────────────────────────────────────────────
    polygon_rpc_urls: str = Field(
        "https://polygon-rpc.com,https://rpc-mainnet.matic.network,"
        "https://matic-mainnet.chainstacklabs.com",
        description="Comma-separated list of RPC endpoints (tried in order)",
    )
    polymarket_chain_id: int = Field(137, description="137 = Polygon mainnet")
    polymarket_host: str = Field(
        "https://clob.polymarket.com", description="Polymarket CLOB API base URL"
    )

    # ── Strategy parameters ───────────────────────────────────────────────────
    kelly_fraction: float = Field(
        0.25, ge=0.01, le=1.0,
        description="Fraction of full Kelly to wager (0.25 = quarter-Kelly)",
    )
    max_position_usd: float = Field(100.0, gt=0, description="Hard cap on any single position in USD")
    min_ev_threshold: float = Field(0.03, description="Minimum expected-value edge required to place an order")
    min_confidence: str = Field("MEDIUM", description="Minimum LLM confidence to act on (LOW | MEDIUM | HIGH)")
    max_open_positions: int = Field(10, gt=0, description="Maximum simultaneous positions")

    # ── Exit / take-profit strategy ───────────────────────────────────────────
    take_profit_pct: float = Field(
        0.15, ge=0.01, le=1.0,
        description="Sell when price rises this fraction above entry (0.15 = +15%)",
    )
    stop_loss_pct: float = Field(
        0.10, ge=0.01, le=1.0,
        description="Sell when price drops this fraction below entry (0.10 = -10%)",
    )
    exit_days_before_end: float = Field(
        1.0, ge=0.0,
        description="Sell open positions this many days before market resolution (avoids binary risk)",
    )

    # ── Market-making (spread capture) ────────────────────────────────────────
    spread_target: float = Field(
        0.02, ge=0.005, le=0.10,
        description="Profit target per round-trip in price units (0.02 = 2 cents)",
    )
    max_mm_markets: int = Field(5, gt=0, description="Max markets to make simultaneously")
    mm_cycle_seconds: int = Field(10, gt=0, description="Fast-loop interval for order management")
    mm_order_size_usd: float = Field(25.0, gt=0, description="Fixed USD size per market-making order")
    max_consecutive_losses: int = Field(3, gt=0, description="Circuit breaker: halt after N consecutive losses")
    min_book_depth_usd: float = Field(500.0, ge=0, description="Minimum order-book depth to trade")

    # ── Bot cadence ───────────────────────────────────────────────────────────
    cycle_interval_seconds: int = Field(300, gt=0, description="Seconds between evaluation cycles")
    dry_run: bool = Field(True, description="When True, evaluate only — no real orders are placed")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default_factory=lambda: _REPO_ROOT / "data",
        description="Root data directory (por defecto: <repo>/data)",
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot_state.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma_storage"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("gemini_api_key")
    @classmethod
    def _validate_gemini_key(cls, v: str) -> str:
        v = (v or "").strip().strip('"').strip("'")
        if not v:
            return v
        if len(v) < 20:
            raise ValueError(
                "GEMINI_API_KEY parece demasiado corta. "
                "Copia la clave completa desde https://aistudio.google.com/app/apikey"
            )
        return v

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, v: str) -> str:
        allowed = {"ollama", "gemini"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return v

    @field_validator("min_confidence")
    @classmethod
    def _validate_confidence(cls, v: str) -> str:
        allowed = {"LOW", "MEDIUM", "HIGH"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"min_confidence must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _ensure_data_dir(self) -> "Settings":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        return self

    # ── Convenience helpers ───────────────────────────────────────────────────
    @property
    def rpc_url_list(self) -> list[str]:
        return [u.strip() for u in self.polygon_rpc_urls.split(",") if u.strip()]

    @property
    def confidence_rank(self) -> dict[str, int]:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    def meets_confidence(self, level: str) -> bool:
        rank = self.confidence_rank
        return rank.get(level.upper(), 0) >= rank.get(self.min_confidence, 1)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_ollama(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
