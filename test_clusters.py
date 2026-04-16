import asyncio
from src.core.config import get_settings
from src.polymarket.clob_client import AsyncClobClient
import re
from collections import defaultdict

async def main():
    cfg = get_settings()
    clob = AsyncClobClient(host=cfg.polymarket_host, private_key="", dry_run=True)
    await clob.initialize()
    markets = await clob.get_markets(limit=500)
    
    # Regex to catch "Will [Asset] be above $[Price] [Time]?" or similar
    # We'll just group by the base question with the numeric part stripped out
    
    clusters = defaultdict(list)
    
    for m in markets:
        if not m.active or m.closed: continue
        q = m.question or ""
        # Find if there's a dollar amount or strike in the question
        m_dollars = re.search(r'\$([0-9,\.]+)', q)
        if m_dollars:
            strike_str = m_dollars.group(1).replace(',', '')
            try:
                strike = float(strike_str)
            except:
                continue
            # Base question: replace the dollar amount with a placeholder
            base_q = q.replace(f'${m_dollars.group(1)}', '$XXX')
            # Group by (base question, endDate)
            end_date = m.days_to_end  # using this as a proxy for end date
            if end_date is None: continue
            
            clusters[(base_q, round(end_date, 3))].append((strike, m))
            
    # Filter clusters with >= 2 strikes
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
    print(f"Found {len(valid_clusters)} valid clusters")
    for k, strikes in list(valid_clusters.items())[:5]:
        print(f"Cluster: {k[0]}")
        for s, m in sorted(strikes):
            print(f"  Strike: {s} | {m.question}")
    
    await clob.close()

asyncio.run(main())
