def get_implied_probability(yes_price: float, no_price: float = None) -> float:
    """
    Extracts the market implied probability from binary market prices.
    If no_price is provided, evaluates the spread midpoint to remove potential bias 
    caused by the spread gap.
    
    Parameters
    ----------
    yes_price : Current best ask or last traded price for YES
    no_price  : Current best ask or last traded price for NO (optional)
    
    Returns
    -------
    Implied probability p_market in range [0.0, 1.0]
    """
    if not (0.0 <= yes_price <= 1.0):
        raise ValueError(f"yes_price must be between 0 and 1, got {yes_price}")
        
    if no_price is not None:
        if not (0.0 <= no_price <= 1.0):
            raise ValueError(f"no_price must be between 0 and 1, got {no_price}")
            
        # The sum of YES + NO prices often slightly exceeds 1.0 due to fees/spread
        # Normalize to find the true midpoint implied probability
        total = yes_price + no_price
        if total > 0:
            return yes_price / total
            
    # Default fallback to direct YES price interpretation
    return yes_price

