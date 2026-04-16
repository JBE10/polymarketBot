import re
from collections import defaultdict
from src.polymarket.models import Market

def get_market_clusters_impl(markets_list):
    clusters = defaultdict(list)
    for m in markets_list:
        if not m.active or m.closed or m.archived: continue
        q = m.question or ""
        m_dollars = re.search(r'\$([0-9,\.]+)', q)
        if m_dollars:
            strike_str = m_dollars.group(1).replace(',', '')
            try:
                strike = float(strike_str)
            except:
                continue
            base_q = q.replace(f'${m_dollars.group(1)}', '$XXX')
            end_date = m.days_to_end
            if end_date is None: continue
            clusters[(base_q, round(end_date, 3))].append((strike, m))
    
    valid_clusters = []
    for (base_q, _), strikes in clusters.items():
        if len(strikes) >= 2:
            # sort by strike ascending
            sorted_strikes = sorted(strikes, key=lambda x: x[0])
            valid_clusters.append({
                "base_question": base_q,
                "markets": sorted_strikes
            })
    return valid_clusters
