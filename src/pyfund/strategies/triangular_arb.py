# src/pyfund/strategies/triangulararb.py

import pandas as pd

from ..data.fetcher import DataFetcher


class TriangularArbitrageStrategy:
    """
    Triangular Arbitrage Strategy for crypto spot markets.
    Example triangle: BTC/USDT → ETH/BTC → ETH/USDT → BTC/USDT
    Detects and quantifies arbitrage opportunities in real-time.
    """

    def __init__(
        self,
        base: str = "BTC",
        quote: str = "USDT",
        intermediate: str = "ETH",
        fee_rate: float = 0.001,  # 0.1% taker fee (Binance spot typical)
        min_profit_bps: float = 5.0,  # Minimum 5 bps profit after fees to trigger
        slippage_bps: float = 1.0,  # Estimated slippage in basis points
    ):
        self.base = base.upper()
        self.quote = quote.upper()
        self.intermediate = intermediate.upper()
        self.fee_rate = fee_rate
        self.min_profit_bps = min_profit_bps
        self.slippage_bps = slippage_bps

        # Define the three trading pairs in the triangle
        self.pairs = [
            f"{self.base}{self.quote}",  # e.g., BTCUSDT
            f"{self.intermediate}{self.base}",  # e.g., ETHBTC
            f"{self.intermediate}{self.quote}",  # e.g., ETHUSDT
        ]

    def _fetch_prices(self) -> dict[str, dict[str | float]]:
        """Fetch latest bid/ask prices for all three pairs"""
        prices = {}
        for pair in self.pairs:
            try:
                ticker = DataFetcher.get_price(pair, period="1d", interval="1m").tail(1)
                # For real crypto exchanges, use order book depth; here we simulate bid/ask
                close = ticker["Close"].iloc[-1]
                bid = close * (1 - 0.0005)  # Simulate bid
                ask = close * (1 + 0.0005)  # Simulate ask
                prices[pair] = {"bid": bid, "ask": ask}
            except Exception as e:
                print(f"Failed to fetch {pair}: {e}")
                return {}
        return prices

    def _forward_arbitrage(self, prices: dict) -> tuple[float | str]:
        """Start with quote currency → try to end with more quote"""
        try:
            # Step 1: Buy intermediate with base (e.g., buy ETH with BTC)
            eth_btc_ask = prices[f"{self.intermediate}{self.base}"]["ask"]
            eth_amount = 1.0 / eth_btc_ask  # How much ETH we get for 1 BTC

            # Step 2: Sell intermediate for quote (e.g., sell ETH for USDT)
            eth_usdt_bid = prices[f"{self.intermediate}{self.quote}"]["bid"]
            usdt_amount = eth_amount * eth_usdt_bid

            # Step 3: Buy base with quote (e.g., buy BTC with USDT)
            btc_usdt_ask = prices[f"{self.base}{self.quote}"]["ask"]
            final_btc = usdt_amount / btc_usdt_ask

            return final_btc, "forward"
        except:
            return 0.0, "forward"

    def _reverse_arbitrage(self, prices: dict) -> tuple[float | str]:
        """Start with quote → opposite direction"""
        try:
            # Reverse path
            btc_usdt_bid = prices[f"{self.base}{self.quote}"]["bid"]
            eth_usdt_ask = prices[f"{self.intermediate}{self.quote}"]["ask"]
            eth_btc_bid = prices[f"{self.intermediate}{self.base}"]["bid"]

            final_btc = (1.0 / btc_usdt_bid) * eth_usdt_ask * eth_btc_bid
            return final_btc, "reverse"
        except:
            return 0.0, "reverse"

    def scan(self) -> dict | None:
        """
        Scan for triangular arbitrage opportunity.
        Returns dict with signal details if profitable, else None.
        """
        prices = self._fetch_prices()
        if not prices:
            return None

        # Test both directions
        forward_final, fwd_dir = self._forward_arbitrage(prices)
        reverse_final, rev_dir = self._reverse_arbitrage(prices)

        final_btc = max(forward_final, reverse_final)
        direction = fwd_dir if forward_final > reverse_final else rev_dir

        gross_profit = final_btc - 1.0
        total_fees = self.fee_rate * 3  # 3 trades
        slippage_cost = self.slippage_bps / 10000
        net_profit = gross_profit - total_fees - slippage_cost

        profit_bps = net_profit * 10000

        if profit_bps >= self.min_profit_bps:
            return {
                "timestamp": pd.Timestamp.now(),
                "triangle": f"{self.base}-{self.intermediate}-{self.quote}",
                "direction": direction,
                "gross_profit_bps": gross_profit * 10000,
                "net_profit_bps": profit_bps,
                "final_amount": final_btc,
                "signal": "BUY" if net_profit > 0 else "NONE",
            }
        return None

    def generate_signals(self, data: pd.DataFrame = None) -> pd.Series:
        """Compatibility with backtester - returns opportunity score"""
        opp = self.scan()
        score = opp["net_profit_bps"] / 100 if opp else 0.0
        return pd.Series([score], index=[pd.Timestamp.now()])


# Quick test when run directly
if __name__ == "__main__":
    arb = TriangularArbitrageStrategy()
    opportunity = arb.scan()
    if opportunity:
        print("🚨 ARBITRAGE OPPORTUNITY FOUND!")
        print(opportunity)
    else:
        print("No triangular arbitrage opportunity right now.")
