"""Parameters and exact loss functions for continuous lead-time demand."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from scipy.special import gammaincc, gamma as gamma_function
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm as lognormal_distribution
from scipy.stats import norm, weibull_min


class LeadTimeDemand(Protocol):
    """Continuous distribution of demand X accumulated over fixed lead time."""

    @property
    def mean(self) -> float: ...

    def isf(self, probability: float) -> float:
        """Find s such that P(X > s) = probability."""
        ...

    def sf(self, threshold: float) -> float:
        """P(X > threshold)."""
        ...

    def expected_shortage(self, threshold: float) -> float:
        """E[max(X-threshold, 0)] (units per replenishment cycle)."""
        ...


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


def _valid_tail(probability: float) -> None:
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("P(X > s) must be strictly between 0 and 1.")


@dataclass(frozen=True, kw_only=True)
class NormalLeadTime:
    """Normal X ~ N(mu, sigma^2). sigma is SD, NOT variance."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("Lead-time mean must be finite.")
        _positive_finite(self.sigma, "Lead-time standard deviation")

    @classmethod
    def from_period_demand(
        cls, *, mean_per_period: float, sd_per_period: float, lead_time_periods: float
    ) -> NormalLeadTime:
        """For independent increments, E[X]=mu*L and Var(X)=sigma^2*L.

        Rates, L, and monetary costs must use a common time unit.
        """
        _positive_finite(mean_per_period, "Mean demand per period")
        _positive_finite(sd_per_period, "SD demand per period")
        _positive_finite(lead_time_periods, "Lead time")
        return cls(
            mu=mean_per_period * lead_time_periods,
            sigma=sd_per_period * math.sqrt(lead_time_periods),
        )

    @property
    def mean(self) -> float:
        return self.mu

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return self.mu + self.sigma * float(norm.isf(probability))

    def sf(self, threshold: float) -> float:
        return float(norm.sf((threshold - self.mu) / self.sigma))

    def expected_shortage(self, threshold: float) -> float:
        k = (threshold - self.mu) / self.sigma
        result = self.sigma * (float(norm.pdf(k)) - k * float(norm.sf(k)))
        # Avoid a tiny negative value from cancellation at very large positive k.
        return max(0.0, result)

    def normal_k(self, threshold: float) -> float:
        return (threshold - self.mu) / self.sigma


@dataclass(frozen=True, kw_only=True)
class UniformLeadTime:
    """Continuous X ~ Uniform(low, high) (demand during lead time)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("Uniform bounds must be finite.")
        if self.high <= self.low or self.low < 0:
            raise ValueError("Require 0 <= low < high for uniform lead-time demand.")

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return self.high - probability * (self.high - self.low)

    def sf(self, threshold: float) -> float:
        if threshold <= self.low:
            return 1.0
        if threshold >= self.high:
            return 0.0
        return (self.high - threshold) / (self.high - self.low)

    def expected_shortage(self, threshold: float) -> float:
        if threshold <= self.low:
            return self.mean - threshold
        if threshold >= self.high:
            return 0.0
        return (self.high - threshold) ** 2 / (2 * (self.high - self.low))


@dataclass(frozen=True, kw_only=True)
class ExponentialLeadTime:
    """Continuous exponential X with mean (scale) in lead-time units."""

    scale: float

    def __post_init__(self) -> None:
        _positive_finite(self.scale, "Exponential lead-time mean")

    @property
    def mean(self) -> float:
        return self.scale

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return -self.scale * math.log(probability)

    def sf(self, threshold: float) -> float:
        if threshold < 0:
            return 1.0
        return math.exp(-threshold / self.scale)

    def expected_shortage(self, threshold: float) -> float:
        if threshold < 0:
            return self.scale - threshold
        return self.scale * math.exp(-threshold / self.scale)


@dataclass(frozen=True, kw_only=True)
class CostParameters:
    """Use one consistent cost/demand time unit (often one year)."""

    demand_rate: float  # D units per accounting period
    order_cost: float  # A money per order
    holding_cost: float  # h money per item per accounting period
    shortage_cost: float  # p money per item short per replenishment cycle
    unit_cost: float = 0.0  # v money per item; additive constant D*v

    def __post_init__(self) -> None:
        for name in ("demand_rate", "order_cost", "holding_cost", "shortage_cost"):
            _positive_finite(getattr(self, name), name)
        if not math.isfinite(self.unit_cost) or self.unit_cost < 0:
            raise ValueError("unit_cost must be finite and >= 0")


# Additional *continuous* distributions.  Discrete laws such as Poisson require
# integer reorder points and finite differences; they deliberately use a
# different optimization model and are not passed to this continuous solver.

@dataclass(frozen=True, kw_only=True)
class GammaLeadTime:
    """Positive and right-skewed lead-time demand X ~ Gamma(shape, scale)."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        _positive_finite(self.shape, "Gamma shape")
        _positive_finite(self.scale, "Gamma scale")

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return float(gamma_distribution.isf(probability, a=self.shape, scale=self.scale))

    def sf(self, threshold: float) -> float:
        if threshold <= 0:
            return 1.0
        return float(gamma_distribution.sf(threshold, a=self.shape, scale=self.scale))

    def expected_shortage(self, threshold: float) -> float:
        if threshold <= 0:
            return self.mean - threshold
        z = threshold / self.scale
        # E[X 1(X>s)] - s P(X>s), via regularized upper incomplete gamma.
        shortage = self.mean * float(gammaincc(self.shape + 1, z)) - (
            threshold * float(gammaincc(self.shape, z))
        )
        return max(0.0, shortage)


@dataclass(frozen=True, kw_only=True)
class LognormalLeadTime:
    """Lognormal X parameterized by its *arithmetic* mean and SD, not log moments."""

    mean_demand: float
    sd_demand: float

    def __post_init__(self) -> None:
        _positive_finite(self.mean_demand, "Lognormal mean")
        _positive_finite(self.sd_demand, "Lognormal standard deviation")

    @property
    def mean(self) -> float:
        return self.mean_demand

    @property
    def sigma_log(self) -> float:
        return math.sqrt(math.log1p((self.sd_demand / self.mean_demand) ** 2))

    @property
    def mu_log(self) -> float:
        return math.log(self.mean_demand) - self.sigma_log**2 / 2

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return float(lognormal_distribution.isf(
            probability, s=self.sigma_log, scale=math.exp(self.mu_log)
        ))

    def sf(self, threshold: float) -> float:
        if threshold <= 0:
            return 1.0
        return float(norm.sf((math.log(threshold) - self.mu_log) / self.sigma_log))

    def expected_shortage(self, threshold: float) -> float:
        if threshold <= 0:
            return self.mean - threshold
        log_s = math.log(threshold)
        k = (log_s - self.mu_log) / self.sigma_log
        first_moment_tail = self.mean * float(norm.sf(k - self.sigma_log))
        return max(0.0, first_moment_tail - threshold * float(norm.sf(k)))


@dataclass(frozen=True, kw_only=True)
class WeibullLeadTime:
    """Nonnegative lead-time demand X ~ Weibull(shape, scale)."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        _positive_finite(self.shape, "Weibull shape")
        _positive_finite(self.scale, "Weibull scale")

    @property
    def mean(self) -> float:
        return self.scale * float(gamma_function(1 + 1 / self.shape))

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        return self.scale * (-math.log(probability)) ** (1 / self.shape)

    def sf(self, threshold: float) -> float:
        if threshold <= 0:
            return 1.0
        return float(weibull_min.sf(threshold, c=self.shape, scale=self.scale))

    def expected_shortage(self, threshold: float) -> float:
        if threshold <= 0:
            return self.mean - threshold
        z = (threshold / self.scale) ** self.shape
        tail_moment = self.mean * float(gammaincc(1 + 1 / self.shape, z))
        return max(0.0, tail_moment - threshold * math.exp(-z))


@dataclass(frozen=True, kw_only=True)
class TriangularLeadTime:
    """Bounded triangular lead-time demand with lower, mode, upper."""

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.low, self.mode, self.high)):
            raise ValueError("Triangular parameters must be finite.")
        if not 0 <= self.low < self.mode < self.high:
            raise ValueError("Triangular demand requires 0 <= low < mode < high.")

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def isf(self, probability: float) -> float:
        _valid_tail(probability)
        right_tail_at_mode = (self.high - self.mode) / (self.high - self.low)
        if probability <= right_tail_at_mode:
            return self.high - math.sqrt(
                probability * (self.high - self.low) * (self.high - self.mode)
            )
        return self.low + math.sqrt(
            (1 - probability) * (self.high - self.low) * (self.mode - self.low)
        )

    def sf(self, threshold: float) -> float:
        if threshold <= self.low:
            return 1.0
        if threshold >= self.high:
            return 0.0
        if threshold < self.mode:
            return 1 - (threshold - self.low) ** 2 / (
                (self.high - self.low) * (self.mode - self.low)
            )
        return (self.high - threshold) ** 2 / (
            (self.high - self.low) * (self.high - self.mode)
        )

    def expected_shortage(self, threshold: float) -> float:
        if threshold <= self.low:
            return self.mean - threshold
        if threshold >= self.high:
            return 0.0
        if threshold < self.mode:
            return self.mean - threshold + (threshold - self.low) ** 3 / (
                3 * (self.high - self.low) * (self.mode - self.low)
            )
        return (self.high - threshold) ** 3 / (
            3 * (self.high - self.low) * (self.high - self.mode)
        )
