"""
Pydantic v2 data models for Polymarket API responses and internal state.
Strict validation prevents silent type coercions at runtime.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Enums ─────────────────────────────────────────────────────────────────────

class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    GTC = "GTC"   # Good-Till-Cancelled
    FOK = "FOK"   # Fill-Or-Kill
    GTD = "GTD"   # Good-Till-Date


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    MATCHED   = "MATCHED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


class Outcome(str, Enum):
    YES = "YES"
    NO  = "NO"


class Confidence(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class Action(str, Enum):
    BUY    = "BUY"
    SKIP   = "SKIP"
    REJECT = "REJECT"


# ── Market ────────────────────────────────────────────────────────────────────

class MarketToken(BaseModel):
    token_id:     str
    outcome:      str
    price:        float = 0.0
    winner:       bool  = False

    @field_validator("price", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> float:
        return float(v) if v is not None else 0.0


class Market(BaseModel):
    """Normalised view of a Polymarket CLOB market."""

    condition_id:     str
    question_id:      str
    question:         str
    description:      str  = ""
    market_slug:      str  = ""
    end_date_iso:     Optional[str]  = None
    start_date_iso:   Optional[str] = None
    active:           bool = True
    closed:           bool = False
    archived:         bool = False
    minimum_order_size: float = 1.0
    minimum_tick_size:  float = 0.01
    tokens:           list[MarketToken] = Field(default_factory=list)
    tags:             list[str]         = Field(default_factory=list)
    volume:           float = 0.0
    volume_24hr:      float = 0.0
    liquidity:        float = 0.0

    @property
    def yes_token(self) -> Optional[MarketToken]:
        for t in self.tokens:
            if t.outcome.upper() == "YES":
                return t
        return self.tokens[0] if self.tokens else None

    @property
    def no_token(self) -> Optional[MarketToken]:
        for t in self.tokens:
            if t.outcome.upper() == "NO":
                return t
        return self.tokens[1] if len(self.tokens) > 1 else None

    @property
    def yes_price(self) -> Optional[float]:
        t = self.yes_token
        return t.price if t else None

    @property
    def no_price(self) -> Optional[float]:
        t = self.no_token
        return t.price if t else None

    @property
    def days_to_end(self) -> Optional[float]:
        if not self.end_date_iso:
            return None
        try:
            end = datetime.fromisoformat(
                self.end_date_iso.replace("Z", "+00:00")
            )
            now = datetime.now(end.tzinfo)
            return (end - now).total_seconds() / 86400
        except Exception:
            return None

    model_config = ConfigDict(populate_by_name=True)


# ── Order Book ────────────────────────────────────────────────────────────────

class PriceLevel(BaseModel):
    price: float
    size:  float

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                "price": float(data.get("price", 0)),
                "size":  float(data.get("size",  0)),
            }
        return data


class OrderBook(BaseModel):
    market:   str = ""
    asset_id: str = ""
    bids:     list[PriceLevel] = Field(default_factory=list)
    asks:     list[PriceLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return max((level.price for level in self.bids), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min((level.price for level in self.asks), default=None)

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth_usd(self, side: Side, levels: int = 5) -> float:
        """Total USD value in the top N levels on one side."""
        book = self.bids if side == Side.BUY else self.asks
        return sum(level.price * level.size for level in book[:levels])


# ── Order ─────────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    token_id:   str
    price:      float   = Field(gt=0, lt=1)
    size:       float   = Field(gt=0, description="Number of shares")
    side:       Side
    order_type: OrderType = OrderType.GTC

    @property
    def cost_usd(self) -> float:
        return self.price * self.size


class OrderResponse(BaseModel):
    order_id:         Optional[str] = None
    status:           OrderStatus   = OrderStatus.PENDING
    transaction_hash: Optional[str] = None
    error_message:    Optional[str] = None
    filled_size:      float = 0.0
    filled_price:     float = 0.0

    @property
    def success(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.MATCHED)


# ── LLM Evaluation result ─────────────────────────────────────────────────────

class MarketEvaluation(BaseModel):
    """
    Structured output from the LLM evaluation step.
    Produced by parsing the function-call JSON returned by the model.
    """

    probability_estimate: float = Field(
        ge=0.0, le=1.0, description="LLM estimate of P(YES)"
    )
    confidence:           Confidence
    reasoning:            str
    key_factors:          list[str] = Field(default_factory=list)
    should_skip:          bool      = False
    skip_reason:          str       = ""

    # ── Computed fields ───────────────────────────────────────────────────────

    def expected_value(self, entry_price: float) -> float:
        """EV per dollar — positive means edge in our favour."""
        return self.probability_estimate - entry_price

    def kelly_fraction(self, entry_price: float, kelly_scale: float = 0.25) -> float:
        """
        Fractional Kelly wager size as fraction of bankroll.
        Returns 0 when there's no edge or the market is mispriced.
        """
        from src.strategy.kelly import kelly_criterion
        return kelly_criterion(
            prob=self.probability_estimate,
            entry_price=entry_price,
            kelly_fraction=kelly_scale,
        )
