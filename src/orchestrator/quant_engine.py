import logging
from typing import Any, List
from datetime import datetime, timezone

from src.storage.database import Database
from src.core.yaml_config import StrategyConfig
from src.storage.models import MarketState, Features, Decision
from src.model.logistic_regression import compute_p_model_up
from src.model.cost_model import get_cost_model, compute_edge_net
from src.portfolio.position_sizing import calculate_dynamic_size, PositionSizingParams
from src.risk.risk_guard import RiskGuard
from src.portfolio.asset_selector import select_top_assets

log = logging.getLogger(__name__)

# ================================
# Temporary Stub Layer
# ================================
class PriceFeedClientStub:
    def get_features(self, asset: str) -> Features:
        # TODO: Replace with CCXT Binance OHLCV adapter
        return Features(
            ema_fast=101.0, 
            ema_slow=100.0, 
            rsi=56.0, 
            momentum=0.4, 
            atr_pctile=55.0
        )
# ================================

class QuantEngine:
    def __init__(self, db: Database, config: StrategyConfig):
        self.db = db
        self.config = config
        self.risk_guard = RiskGuard(db=db, config=self.config.risk)
        self.price_feed = PriceFeedClientStub()

    def _is_in_no_trade_window(self, expiry_utc: datetime) -> bool:
        """Blocks execution if approaching rapidly resolving state."""
        now = datetime.now(timezone.utc)
        time_limit_sec = self.config.general.no_trade_last_minutes_before_expiry * 60
        return (expiry_utc - now).total_seconds() <= time_limit_sec

    def _passes_market_filters(self, m: MarketState) -> bool:
        """Enforces configured spread widths and active USD depth requirements."""
        max_spread = self.config.market_filters.max_spread_pct.get(m.asset, 1.5)
        min_depth = self.config.market_filters.min_orderbook_depth_usd.get(m.asset, 1500)
        
        if m.spread_pct > max_spread:
            return False
            
        if m.depth_usd < min_depth:
            return False
            
        return True

    def _choose_side_and_edge(self, p_model_up: float, p_market_up: float, asset: str):
        """Compares strictly orthogonal vectors (UP vs DOWN) determining maximum theoretically viable return."""
        cost_cfg = get_cost_model(asset, self.config)
        edge_up = compute_edge_net(p_model_up, p_market_up, "UP", cost_cfg)
        edge_down = compute_edge_net(p_model_up, p_market_up, "DOWN", cost_cfg)
        
        if edge_up >= edge_down:
            return "UP", edge_up, cost_cfg
        return "DOWN", edge_down, cost_cfg

    def _passes_trend_alignment(self, side: str, feat: Features) -> bool:
        """Apply deterministic trend alignment gate from config semantics."""
        if not self.config.entry_rules.require_trend_alignment:
            return True

        if side == "UP":
            return feat.ema_fast > feat.ema_slow and feat.rsi > 52
        return feat.ema_fast < feat.ema_slow and feat.rsi < 48

    def _passes_noise_filter(self, feat: Features) -> bool:
        """Block low-quality volatility regimes when configured."""
        if not self.config.entry_rules.block_if_high_noise:
            return True

        min_atr_pctile = self.config.entry_rules.high_noise_rules.get("min_atr_percentile", 0.0)
        return feat.atr_pctile >= float(min_atr_pctile)

    @staticmethod
    def _trend_quality(side: str, feat: Features) -> float:
        """Trend quality in [0, 1] used by the ranker."""
        if side == "UP":
            return 1.0 if (feat.ema_fast > feat.ema_slow and feat.rsi > 52) else 0.6
        return 1.0 if (feat.ema_fast < feat.ema_slow and feat.rsi < 48) else 0.6

    @staticmethod
    def _regime_from_atr_pctile(atr_pctile: float | None) -> str:
        if atr_pctile is None:
            return "MID_VOL"
        if atr_pctile < 33:
            return "LOW_VOL"
        if atr_pctile > 66:
            return "HIGH_VOL"
        return "MID_VOL"

    async def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        insert_fn = getattr(self.db, "insert_decision_snapshot", None)
        if not callable(insert_fn):
            return
        try:
            await insert_fn(**snapshot)
        except Exception as exc:
            log.warning("Could not persist decision snapshot: %s", exc)

    async def evaluate_round(self, markets: List[MarketState]) -> List[Decision]:
        """
        Orchestrates the quantitative extraction pipeline dynamically loading configuration limits.
        """
        evaluations: List[Decision] = []
        evaluation_snapshots: list[tuple[Decision, dict[str, Any]]] = []
        enabled_assets = set(self.config.universe.enabled_assets)
        
        for m in markets:
            snapshot: dict[str, Any] = {
                "asset": m.asset,
                "market_id": m.market_id,
                "side": "N/A",
                "action": "SKIP",
                "reason_code": "UNPROCESSED",
                "regime": "MID_VOL",
                "p_model_up": None,
                "p_market_up": m.p_market_up,
                "edge_net_pct": None,
                "score": None,
                "threshold_pct": None,
                "total_cost_pct": None,
                "fee_pct": None,
                "slippage_pct": None,
                "latency_buffer_pct": None,
                "notional_usd": None,
                "spread_pct": m.spread_pct,
                "depth_usd": m.depth_usd,
                "expiry_utc": m.expiry_utc.isoformat(),
                "feature_ema_fast": None,
                "feature_ema_slow": None,
                "feature_rsi": None,
                "feature_momentum": None,
                "feature_atr_pctile": None,
                "meta": {},
            }

            if m.asset not in enabled_assets:
                snapshot["reason_code"] = "ASSET_NOT_ENABLED"
                await self._persist_snapshot(snapshot)
                continue

            if self._is_in_no_trade_window(m.expiry_utc):
                snapshot["reason_code"] = "NO_TRADE_WINDOW"
                await self._persist_snapshot(snapshot)
                continue
                
            if not self._passes_market_filters(m):
                snapshot["reason_code"] = "MARKET_FILTER_FAIL"
                await self._persist_snapshot(snapshot)
                continue

            # Check Hard Drawdown Risk Guards before analyzing technicals
            bankroll = self.config.risk.bankroll_usd
            if not await self.risk_guard.is_safe_to_trade(bankroll, m.asset):
                snapshot["reason_code"] = "RISK_GUARD_BLOCK"
                await self._persist_snapshot(snapshot)
                continue

            # Retrieve Indicator Features (Stubbed to map exact algorithm logic)
            feat = self.price_feed.get_features(m.asset)
            snapshot["feature_ema_fast"] = feat.ema_fast
            snapshot["feature_ema_slow"] = feat.ema_slow
            snapshot["feature_rsi"] = feat.rsi
            snapshot["feature_momentum"] = feat.momentum
            snapshot["feature_atr_pctile"] = feat.atr_pctile
            snapshot["regime"] = self._regime_from_atr_pctile(feat.atr_pctile)

            if not self._passes_noise_filter(feat):
                snapshot["reason_code"] = "HIGH_NOISE_BLOCK"
                await self._persist_snapshot(snapshot)
                continue

            p_model_up = compute_p_model_up(feat, self.config.model)
            snapshot["p_model_up"] = p_model_up

            # Evaluate execution directions
            side, edge_net, cost_cfg = self._choose_side_and_edge(p_model_up, m.p_market_up, m.asset)
            snapshot["side"] = side
            snapshot["threshold_pct"] = cost_cfg.theta * 100.0
            snapshot["total_cost_pct"] = cost_cfg.total_cost * 100.0
            snapshot["fee_pct"] = cost_cfg.fee * 100.0
            snapshot["slippage_pct"] = cost_cfg.slippage * 100.0
            snapshot["latency_buffer_pct"] = self.config.cost_model.extra_buffer_pct
            snapshot["edge_net_pct"] = edge_net * 100.0

            if not self._passes_trend_alignment(side, feat):
                snapshot["reason_code"] = "TREND_MISALIGNMENT"
                await self._persist_snapshot(snapshot)
                continue
            
            # Entry condition cutoff
            if edge_net < cost_cfg.theta:
                snapshot["reason_code"] = "EDGE_BELOW_THRESHOLD"
                await self._persist_snapshot(snapshot)
                continue

            # Generate dynamic ranking score matching the reference weighting 
            # score = edge_net * liquidity_score * trend_quality * (1 - spread_penalty)
            min_depth = self.config.market_filters.min_orderbook_depth_usd.get(m.asset, 1500.0)
            liquidity_score = min(1.0, m.depth_usd / (min_depth * 2))
            spread_penalty = min(0.9, m.spread_pct / 10.0)
            trend_quality = self._trend_quality(side, feat)

            # Convert edge to percentage points for ranking consistency with configured score floor.
            edge_points = edge_net * 100.0
            score = edge_points * liquidity_score * trend_quality * (1.0 - spread_penalty)
            snapshot["score"] = score
            
            # Require minimum scoring cutoff
            if score < self.config.ranking.min_score_to_trade:
                snapshot["reason_code"] = "RANK_SCORE_BELOW_MIN"
                await self._persist_snapshot(snapshot)
                continue

            # Position Sizing
            sizing_params = PositionSizingParams(risk_cfg=self.config.risk, sizing_cfg=self.config.position_sizing)
            size = calculate_dynamic_size(
                params=sizing_params,
                edge_net=edge_net,
                theta_asset=cost_cfg.theta
            )
            
            if size <= 0:
                snapshot["reason_code"] = "SIZE_BELOW_MIN_NOTIONAL"
                await self._persist_snapshot(snapshot)
                continue
            snapshot["notional_usd"] = size
                
            decision = Decision(
                asset=m.asset,
                side=side,
                p_model_up=round(p_model_up, 4),
                p_market_up=round(m.p_market_up, 4),
                edge_net_pct=round(edge_net * 100, 3), # Store internally as percentage visualization
                score=round(score, 4),
                notional_usd=size,
                reason=f"edge_net({edge_net:.3f}) > theta({cost_cfg.theta:.3f}) | Valid Constraints"
            )
            evaluations.append(decision)
            snapshot["action"] = "CANDIDATE"
            snapshot["reason_code"] = "BUY_CANDIDATE"
            evaluation_snapshots.append((decision, snapshot))

        # Select Top Assets applying generic maximums (e.g. max 2 markets globally executed per cycle)
        max_assets = self.config.universe.max_assets_per_round
        final_picks = select_top_assets(evaluations, top_n=max_assets)
        selected_ids = {id(p) for p in final_picks}

        for decision, snapshot in evaluation_snapshots:
            if id(decision) in selected_ids:
                snapshot["action"] = "BUY"
                snapshot["reason_code"] = "SELECTED_TOP_N"
            else:
                snapshot["action"] = "SKIP"
                snapshot["reason_code"] = "NOT_IN_TOP_N"
                snapshot["notional_usd"] = 0.0
            await self._persist_snapshot(snapshot)
        
        for pick in final_picks:
            log.info(
                "APPROVED QUANT TRADE | Activo: %s | Mercado: %s | p_model_up: %.3f | p_market_up: %.3f | Side: %s | Edge: %.2f%% | Sizing: $%.2f",
                pick.asset, pick.asset[-8:], pick.p_model_up, pick.p_market_up, pick.side, pick.edge_net_pct, pick.notional_usd
            )
            
        return final_picks
