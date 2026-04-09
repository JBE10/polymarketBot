#!/usr/bin/env python3
"""
Polymarket Intelligence — Advanced Terminal Analytics Dashboard
Run: python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ui.app import PolymarketApp


def main() -> None:
    app = PolymarketApp()
    app.run()


if __name__ == "__main__":
    main()
