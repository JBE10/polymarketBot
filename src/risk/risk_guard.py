import logging
from src.core.yaml_config import RiskConfig

log = logging.getLogger(__name__)

class RiskGuard:
    def __init__(self, db, config: RiskConfig):
        """
        Parameters
        ----------
        db     : Database instance configured to track daily stats and recent trades
        config : RiskConfig derived from the YAML schema
        """
        self.db = db
        self.config = config
        
        self.max_total_exposure = config.max_total_exposure_pct / 100.0
        self.max_daily_dd = config.max_daily_drawdown_pct / 100.0
        
        # Cooldown rules
        cooldown_cfg = config.consecutive_losses_cooldown
        self.cooldown_enabled = cooldown_cfg.get("enabled", True)
        self.cooldown_losses = cooldown_cfg.get("losses", 2)

    async def is_safe_to_trade(self, bankroll: float, asset_symbol: str) -> bool:
        """
        Runs portfolio-level guards before entering a trade.
        Returns True if safe, False if blocked by risk guard.
        """
        if bankroll <= 0:
            return False

        # 1. Kill switch: Max Daily Drawdown
        daily_pnl = await self.db.get_daily_mm_pnl()  # Reuse helper function
        if daily_pnl < 0:
            dd_pct = abs(daily_pnl) / bankroll
            if dd_pct >= self.max_daily_dd:
                log.warning(f"RiskGuard Blocked: Daily Drawdown limit hit ({dd_pct:.2%} >= {self.max_daily_dd:.2%})")
                return False

        # 2. Max Total Exposure
        # Get total open position USD
        open_positions = await self.db.get_open_positions()
        total_exposure_usd = sum(p.get("size_usd", 0) for p in open_positions)
        exposure_pct = total_exposure_usd / bankroll
        
        if exposure_pct >= self.max_total_exposure:
            log.warning(f"RiskGuard Blocked: Total exposure limit hit ({exposure_pct:.2%} >= {self.max_total_exposure:.2%})")
            return False

        # 3. Asset Cooldown (Consecutive losses)
        if self.cooldown_enabled:
            # 1 pool cooldown required if N consecutive losses for this asset
            consecutive_losses = await self.db.get_consecutive_losses(asset_symbol)
            if consecutive_losses >= self.cooldown_losses:
                log.warning(f"RiskGuard Blocked: Cooldown active for {asset_symbol} due to >= {self.cooldown_losses} consecutive losses")
                return False

        return True

    def validate_time_filter(self, minutes_to_end: float, min_required: float = 3.0) -> bool:
        """
        Avoids trading in the tail-end volatile period of a pool.
        """
        if minutes_to_end is not None and minutes_to_end <= min_required:
            log.info(f"RiskGuard Blocked: {minutes_to_end}m to resolution is beneath the {min_required}m threshold")
            return False
        return True
