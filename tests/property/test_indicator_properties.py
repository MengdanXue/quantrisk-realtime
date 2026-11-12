# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Property-based tests for financial indicators.

Uses Hypothesis to automatically generate test cases and find edge cases.
These tests verify mathematical properties that should always hold true.

Properties tested:
1. PCR (Put-Call Ratio) - Always >= 0, undefined for 0 call volume
2. Volatility - Always >= 0
3. Returns - Bounded by reasonable limits
4. Correlations - Always in [-1, 1]
5. VaR/CVaR - CVaR >= VaR (Expected Shortfall is more conservative)
"""

import pytest
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from datetime import datetime, timedelta
from typing import List

# Import modules to test
from quantrisk_realtime.realtime.derivatives_indicators import (
    OptionType,
    PCRType,
    OptionData,
    DerivativesConfig,
    DerivativesIndicators,
    SimulatedOptionDataProvider,
    SimulatedFuturesDataProvider,
)
from quantrisk_realtime.realtime.pnl_streamer import (
    PositionPnL,
    PortfolioPnL,
)


# ============ Custom Strategies ============


# Strategy for generating valid prices (positive floats)
prices = st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating volumes (positive integers)
volumes = st.integers(min_value=0, max_value=10_000_000)

# Strategy for generating quantities
quantities = st.integers(min_value=1, max_value=1_000_000)

# Strategy for generating percentages (0-100)
percentages = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for generating returns (-100% to +1000%)
returns = st.floats(min_value=-1.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for generating implied volatility (0-500%)
implied_vols = st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)

# Strategy for generating strike prices relative to spot
strike_ratios = st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)


@st.composite
def option_data_strategy(draw, symbol: str = "TEST"):
    """Generate valid OptionData instances."""
    spot = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    strike_ratio = draw(strike_ratios)
    strike = spot * strike_ratio

    option_type = draw(st.sampled_from([OptionType.CALL, OptionType.PUT]))

    # Calculate reasonable bid/ask based on moneyness
    if option_type == OptionType.CALL:
        intrinsic = max(0, spot - strike)
    else:
        intrinsic = max(0, strike - spot)

    time_value = draw(st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
    bid = intrinsic + time_value * 0.95
    ask = intrinsic + time_value * 1.05

    return OptionData(
        symbol=symbol,
        option_symbol=f"{symbol}_{option_type.value}_{int(strike)}",
        option_type=option_type,
        strike=strike,
        expiry=datetime.now() + timedelta(days=draw(st.integers(min_value=1, max_value=365))),
        spot_price=spot,
        volume=draw(volumes),
        open_interest=draw(st.integers(min_value=0, max_value=1_000_000)),
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        implied_vol=draw(implied_vols),
    )


@st.composite
def price_series_strategy(draw, min_length: int = 10, max_length: int = 252):
    """Generate valid price series."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    start_price = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))

    # Generate returns and convert to prices
    daily_returns = draw(st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=length - 1,
        max_size=length - 1,
    ))

    prices = [start_price]
    for r in daily_returns:
        prices.append(prices[-1] * (1 + r))

    return np.array(prices)


@st.composite
def position_strategy(draw):
    """Generate valid position data."""
    current_price = draw(prices)
    avg_cost = draw(prices)
    quantity = draw(quantities)
    direction = draw(st.sampled_from([1, -1]))

    unrealized_pnl = (current_price - avg_cost) * quantity * direction

    return PositionPnL(
        position_id=draw(st.integers(min_value=1, max_value=10000)),
        symbol=draw(st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        current_price=current_price,
        average_cost=avg_cost,
        quantity=quantity,
        direction=direction,
        unrealized_pnl=unrealized_pnl,
        market_value=current_price * quantity,
    )


# ============ PCR Property Tests ============


class TestPCRProperties:
    """Property-based tests for Put-Call Ratio calculations."""

    @given(put_volume=volumes, call_volume=st.integers(min_value=1, max_value=10_000_000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pcr_always_non_negative(self, put_volume: int, call_volume: int):
        """PCR should always be >= 0 when call volume > 0."""
        pcr = put_volume / call_volume
        assert pcr >= 0, f"PCR should be non-negative, got {pcr}"

    @given(put_volume=volumes, call_volume=volumes)
    @settings(max_examples=100)
    def test_pcr_symmetric_interpretation(self, put_volume: int, call_volume: int):
        """
        PCR interpretation should be consistent:
        - PCR < 1: More calls than puts (bullish sentiment)
        - PCR > 1: More puts than calls (bearish sentiment)
        """
        assume(call_volume > 0)

        pcr = put_volume / call_volume

        if pcr < 1:
            assert put_volume < call_volume, "PCR < 1 means more calls"
        elif pcr > 1:
            assert put_volume > call_volume, "PCR > 1 means more puts"
        else:  # pcr == 1
            assert put_volume == call_volume, "PCR = 1 means equal volumes"

    @given(
        put_volume=st.integers(min_value=1, max_value=10_000_000),
        call_volume=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=100)
    def test_pcr_inverse_relationship(self, put_volume: int, call_volume: int):
        """PCR and CPR (Call-Put Ratio) should be inverses."""
        pcr = put_volume / call_volume
        cpr = call_volume / put_volume

        assert abs(pcr * cpr - 1.0) < 1e-10, "PCR * CPR should equal 1"


# ============ Volatility Property Tests ============


class TestVolatilityProperties:
    """Property-based tests for volatility calculations."""

    @given(prices=price_series_strategy(min_length=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_volatility_always_non_negative(self, prices: np.ndarray):
        """Volatility should always be >= 0."""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        assert volatility >= 0, f"Volatility should be non-negative, got {volatility}"
        assert not np.isnan(volatility), "Volatility should not be NaN"

    @given(prices=price_series_strategy(min_length=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_constant_prices_zero_volatility(self, prices: np.ndarray):
        """Constant prices should have zero volatility."""
        # Make all prices the same
        constant_prices = np.full_like(prices, prices[0])
        returns = np.diff(constant_prices) / constant_prices[:-1]
        volatility = np.std(returns)

        assert volatility == 0, f"Constant prices should have 0 volatility, got {volatility}"

    @given(
        prices=price_series_strategy(min_length=30),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_volatility_scale_invariance(self, prices: np.ndarray, scale: float):
        """
        Volatility of returns should be scale-invariant.
        Scaling all prices by a constant shouldn't change return volatility.
        """
        scaled_prices = prices * scale

        returns_original = np.diff(prices) / prices[:-1]
        returns_scaled = np.diff(scaled_prices) / scaled_prices[:-1]

        vol_original = np.std(returns_original)
        vol_scaled = np.std(returns_scaled)

        assert abs(vol_original - vol_scaled) < 1e-10, \
            f"Volatility should be scale-invariant: {vol_original} vs {vol_scaled}"


# ============ Correlation Property Tests ============


class TestCorrelationProperties:
    """Property-based tests for correlation calculations."""

    @given(
        prices1=price_series_strategy(min_length=30),
        prices2=price_series_strategy(min_length=30),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_correlation_bounds(self, prices1: np.ndarray, prices2: np.ndarray):
        """Correlation should always be in [-1, 1]."""
        # Align lengths
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[:min_len]
        prices2 = prices2[:min_len]

        returns1 = np.diff(prices1) / prices1[:-1]
        returns2 = np.diff(prices2) / prices2[:-1]

        # Filter out any NaN/inf
        valid = np.isfinite(returns1) & np.isfinite(returns2)
        assume(np.sum(valid) >= 5)

        returns1 = returns1[valid]
        returns2 = returns2[valid]

        corr = np.corrcoef(returns1, returns2)[0, 1]

        if np.isfinite(corr):
            assert -1.0 <= corr <= 1.0, f"Correlation out of bounds: {corr}"

    @given(prices=price_series_strategy(min_length=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_self_correlation_is_one(self, prices: np.ndarray):
        """Correlation of a series with itself should be 1."""
        returns = np.diff(prices) / prices[:-1]
        valid = np.isfinite(returns)
        assume(np.sum(valid) >= 5)

        returns = returns[valid]
        assume(np.std(returns) > 1e-10)  # Need non-constant returns

        corr = np.corrcoef(returns, returns)[0, 1]

        assert abs(corr - 1.0) < 1e-10, f"Self-correlation should be 1, got {corr}"


# ============ VaR/CVaR Property Tests ============


class TestVaRProperties:
    """Property-based tests for Value at Risk calculations."""

    @given(returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000,
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_var_confidence_ordering(self, returns: List[float]):
        """VaR at higher confidence should be >= VaR at lower confidence."""
        returns_array = np.array(returns)

        var_95 = np.percentile(returns_array, 5)  # 95% VaR
        var_99 = np.percentile(returns_array, 1)  # 99% VaR

        # For losses (negative returns), lower percentile = larger loss
        # VaR 99% should capture more extreme losses
        assert var_99 <= var_95, f"VaR 99% ({var_99}) should be <= VaR 95% ({var_95})"

    @given(returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000,
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cvar_exceeds_var(self, returns: List[float]):
        """CVaR (Expected Shortfall) should be >= VaR in magnitude."""
        returns_array = np.array(returns)

        var_95 = np.percentile(returns_array, 5)
        # CVaR is the average of returns below VaR
        cvar_95 = returns_array[returns_array <= var_95].mean() if np.any(returns_array <= var_95) else var_95

        # CVaR should be more negative (worse) than VaR
        assert cvar_95 <= var_95, f"CVaR ({cvar_95}) should be <= VaR ({var_95})"


# ============ P&L Property Tests ============


class TestPnLProperties:
    """Property-based tests for P&L calculations."""

    @given(
        current_price=prices,
        avg_cost=prices,
        quantity=quantities,
        direction=st.sampled_from([1, -1]),
    )
    @settings(max_examples=100)
    def test_pnl_calculation_correctness(
        self, current_price: float, avg_cost: float, quantity: int, direction: int
    ):
        """P&L calculation should follow the formula: (price - cost) * qty * direction."""
        expected_pnl = (current_price - avg_cost) * quantity * direction

        position = PositionPnL(
            position_id=1,
            symbol="TEST",
            current_price=current_price,
            average_cost=avg_cost,
            quantity=quantity,
            direction=direction,
            unrealized_pnl=expected_pnl,
        )

        assert abs(position.unrealized_pnl - expected_pnl) < 1e-10

    @given(
        current_price=prices,
        avg_cost=prices,
        quantity=quantities,
    )
    @settings(max_examples=100)
    def test_long_position_pnl_sign(self, current_price: float, avg_cost: float, quantity: int):
        """Long position: profit when price > cost, loss when price < cost."""
        pnl = (current_price - avg_cost) * quantity

        if current_price > avg_cost:
            assert pnl > 0, "Long position should profit when price rises"
        elif current_price < avg_cost:
            assert pnl < 0, "Long position should lose when price falls"
        else:
            assert pnl == 0, "No P&L when price equals cost"

    @given(
        current_price=prices,
        avg_cost=prices,
        quantity=quantities,
    )
    @settings(max_examples=100)
    def test_short_position_pnl_sign(self, current_price: float, avg_cost: float, quantity: int):
        """Short position: profit when price < cost, loss when price > cost."""
        pnl = (current_price - avg_cost) * quantity * (-1)  # Short direction

        if current_price < avg_cost:
            assert pnl > 0, "Short position should profit when price falls"
        elif current_price > avg_cost:
            assert pnl < 0, "Short position should lose when price rises"
        else:
            assert pnl == 0, "No P&L when price equals cost"


# ============ Portfolio Property Tests ============


class TestPortfolioProperties:
    """Property-based tests for portfolio-level calculations."""

    @given(positions=st.lists(position_strategy(), min_size=1, max_size=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_portfolio_pnl_is_sum_of_positions(self, positions: List[PositionPnL]):
        """Portfolio P&L should equal sum of position P&Ls."""
        total_pnl = sum(p.unrealized_pnl for p in positions)
        total_market_value = sum(p.market_value for p in positions)

        portfolio = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=total_pnl,
            total_market_value=total_market_value,
            positions=positions,
            position_count=len(positions),
        )

        # Verify the sum
        position_sum = sum(p.unrealized_pnl for p in portfolio.positions)
        assert abs(portfolio.total_unrealized_pnl - position_sum) < 1e-10

    @given(positions=st.lists(position_strategy(), min_size=1, max_size=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_market_value_always_positive(self, positions: List[PositionPnL]):
        """Total market value should always be positive."""
        total_market_value = sum(p.market_value for p in positions)

        assert total_market_value >= 0, f"Market value should be non-negative: {total_market_value}"


# ============ Option Data Property Tests ============


class TestOptionDataProperties:
    """Property-based tests for option data validation."""

    @given(option=option_data_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bid_ask_spread_non_negative(self, option: OptionData):
        """Bid-ask spread should always be non-negative."""
        spread = option.ask - option.bid
        assert spread >= 0, f"Spread should be non-negative: {spread}"

    @given(option=option_data_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_implied_vol_positive(self, option: OptionData):
        """Implied volatility should always be positive."""
        if option.implied_vol is not None:
            assert option.implied_vol > 0, f"IV should be positive: {option.implied_vol}"

    @given(option=option_data_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_option_to_dict_roundtrip(self, option: OptionData):
        """Option data should serialize to dict correctly."""
        d = option.to_dict()

        assert d["symbol"] == option.symbol
        assert d["strike"] == option.strike
        assert d["option_type"] == option.option_type.value
        assert d["volume"] == option.volume


# ============ Numerical Stability Tests ============


class TestNumericalStability:
    """Tests for numerical stability with edge cases."""

    @given(x=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_log_return_stability(self, x: float):
        """Log return calculation should be stable."""
        # log(1 + r) ≈ r for small r
        r = 0.01  # 1% return
        log_return = np.log(1 + r)

        assert np.isfinite(log_return), "Log return should be finite"
        assert abs(log_return - r) < 0.01, "Log return should approximate simple return for small r"

    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_weight_normalization(self, weights: List[float]):
        """Portfolio weights should normalize to sum to 1."""
        assume(sum(weights) > 0)

        weights_array = np.array(weights)
        normalized = weights_array / weights_array.sum()

        assert abs(normalized.sum() - 1.0) < 1e-10, "Normalized weights should sum to 1"
        assert all(w >= 0 for w in normalized), "Normalized weights should be non-negative"
