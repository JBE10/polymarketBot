from typing import List
from src.storage.models import Decision

def select_top_assets(decisions: List[Decision], top_n: int = 1) -> List[Decision]:
    """
    Ranks the approved trading decisions relying on the composited execution score,
    which scales explicit edge magnitude against liquidity strength and spread penalties.
    
    Returns the top_n items to avoid over-trading concurrent cycles.
    """
    # Sort strictly by derived numerical score, descending
    decisions.sort(key=lambda d: d.score, reverse=True)
    
    return decisions[:top_n]
