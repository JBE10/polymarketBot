"""
AsyncWeb3 provider with transparent RPC failover.

Multiple Polygon RPC endpoints are tried in order; the first one that
responds with is_connected() == True is used.  A background health-check
task periodically rotates to a healthier endpoint if the active one degrades.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider  # available in both web3 6.x and 7.x

log = logging.getLogger(__name__)

# Polygon-native USDC (PoS bridged) and CTF Exchange contract addresses
POLYGON_USDC  = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE  = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF  = "0xd91E80cF2Ed145c2a4Bb5EFD32b60eDd7D1f1AaA"


class PolyProvider:
    """
    Thin async wrapper around AsyncWeb3 with built-in RPC failover.

    Usage
    -----
        provider = await PolyProvider.create(["https://polygon-rpc.com", ...])
        block = await provider.w3.eth.block_number
        await provider.close()
    """

    def __init__(self, w3: AsyncWeb3, url: str, all_urls: list[str]) -> None:
        self.w3       = w3
        self.active   = url
        self._all     = all_urls
        self._hc_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, rpc_urls: list[str]) -> "PolyProvider":
        """
        Try each URL in order and return a provider backed by the first
        reachable endpoint.  Raises RuntimeError if none connect.
        """
        for url in rpc_urls:
            w3: AsyncWeb3 | None = None
            try:
                w3 = AsyncWeb3(AsyncHTTPProvider(url))
                if await w3.is_connected():
                    log.info("Connected to RPC: %s", url)
                    inst = cls(w3, url, rpc_urls)
                    inst._hc_task = asyncio.create_task(inst._health_loop())
                    return inst
            except Exception as exc:
                log.warning("RPC %s unreachable: %s", url, exc)
            finally:
                if w3 is not None:
                    await w3.provider.disconnect()

        raise RuntimeError(
            f"No reachable Polygon RPC endpoint among: {rpc_urls}"
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    async def get_block_number(self) -> int:
        return await self.w3.eth.block_number

    async def get_eth_balance(self, address: str) -> float:
        """Return MATIC balance in Ether units."""
        raw = await self.w3.eth.get_balance(self.w3.to_checksum_address(address))
        return float(self.w3.from_wei(raw, "ether"))

    async def get_usdc_balance(self, address: str) -> float:
        """Return USDC balance (6 decimals)."""
        abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function",
            }
        ]
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(POLYGON_USDC), abi=abi
        )
        raw: int = await contract.functions.balanceOf(
            self.w3.to_checksum_address(address)
        ).call()
        return raw / 1_000_000  # USDC has 6 decimals

    async def is_alive(self) -> bool:
        try:
            await self.w3.eth.block_number
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._hc_task and not self._hc_task.done():
            self._hc_task.cancel()
            try:
                await self._hc_task
            except asyncio.CancelledError:
                pass

    # ── Background health check ───────────────────────────────────────────────

    async def _health_loop(self) -> None:
        """Every 60 s, verify the current endpoint; rotate to backup if dead."""
        while True:
            await asyncio.sleep(60)
            try:
                if not await self.is_alive():
                    log.warning("RPC %s is unresponsive, rotating…", self.active)
                    await self._rotate()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("Health-check error: %s", exc)

    async def _rotate(self) -> None:
        """Switch to the next reachable RPC endpoint."""
        remaining = [u for u in self._all if u != self.active]
        for url in remaining:
            try:
                w3 = AsyncWeb3(AsyncHTTPProvider(url))
                if await w3.is_connected():
                    self.w3    = w3
                    self.active = url
                    log.info("Rotated to RPC: %s", url)
                    return
            except Exception:
                continue
        log.error("All RPC endpoints are unresponsive.")
