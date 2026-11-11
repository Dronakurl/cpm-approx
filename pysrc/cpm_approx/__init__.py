"""Analytical VaR and expected shortfall of credit portfolios in the multi-factor Merton model.

Implements the multi-factor adjustment of Pykhtin (Risk, March 2004). All figures are loss rates relative
to total exposure; the loss quantile includes the expected loss.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native

__all__ = ["RiskResult", "analyze", "paper_portfolio", "portfolio_risk"]
__version__ = "0.1.0"


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Result of :func:`portfolio_risk` (loss rates)."""

    expected_loss: float
    var_limit: float  #: comparable one-factor portfolio (infinitely many small facilities)
    var_inf: float  #: plus the multi-factor adjustment
    var: float  #: plus the granularity adjustment (the final VaR)
    es_limit: float
    es: float  #: expected shortfall


def portfolio_risk(
    pd: Sequence[float],
    lgd: Sequence[float],
    exposure: Sequence[float],
    asset_correlation: Sequence[float],
    confidence: float,
    *,
    factor_weights: Sequence[Sequence[float]] | None = None,
    factor_correlation: Sequence[Sequence[float]] | None = None,
    loadings: Sequence[Sequence[float]] | None = None,
    lgd_std: Sequence[float] | None = None,
) -> RiskResult:
    """VaR and expected shortfall of a portfolio.

    One entry per facility: ``pd`` (default probability), ``lgd`` (mean loss given default), ``exposure``,
    ``asset_correlation`` (the squared loading R² on the systematic factors), optional ``lgd_std``.

    Factor structure, either ``factor_weights`` (sector weights per facility, optionally with
    ``factor_correlation``, a matrix of the factors) or ``loadings`` (rows of unit norm on independent
    standard normal factors). Without both, a single factor is used. NumPy arrays are accepted.
    Raises ``ValueError`` for invalid input. The GIL is released during the computation.
    """
    return RiskResult(*_native.risk(
        _lists(pd), _lists(lgd), _lists(exposure), _lists(asset_correlation), confidence,
        _lists(factor_weights), _lists(factor_correlation), _lists(loadings), _lists(lgd_std),
    ))


def paper_portfolio(rho: float = 0.3, counts: Sequence[int] | None = None) -> dict[str, Any]:
    """Example portfolio of Pykhtin (2004), Table C: 11 factors, 10 buckets.

    ``rho`` is the correlation between the industry factors (0.1, 0.3, 0.5 in the paper), ``counts`` the number
    of facilities per bucket (default: portfolio I, 750 facilities). Usage:
    ``portfolio_risk(**paper_portfolio(0.3), confidence=0.999)``.
    """
    pd, lgd, exposure, r2, lgd_std, loadings = _native.paper_portfolio(
        rho, list(counts if counts is not None else _native.COUNTS_I))
    return {"pd": pd, "lgd": lgd, "exposure": exposure, "asset_correlation": r2,
            "lgd_std": lgd_std, "loadings": loadings}


def analyze(request: Mapping[str, Any]) -> dict[str, Any]:
    """Same request and response format as the ``cpm-approx`` command line tool (JSON-like dictionaries)."""
    obligors = request.get("obligors") or []
    if not obligors:
        raise ValueError("obligors must not be empty")
    corr = request.get("factor_correlation") or None
    nf = len(corr) if corr else 1
    weights = []
    for i, o in enumerate(obligors):
        w = list(o.get("factor_weights") or [])
        if len(w) > nf:
            raise ValueError(f"obligor {i}: {len(w)} factor weights but {nf} factors")
        weights.append(w + [0.0] * (nf - len(w)) if w else [1.0] + [0.0] * (nf - 1))
    q = float(request["confidence_level"])
    r = portfolio_risk(
        [o["pd"] for o in obligors], [o["lgd"] for o in obligors], [o["exposure"] for o in obligors],
        [o["asset_correlation"] for o in obligors], q, factor_weights=weights, factor_correlation=corr,
        lgd_std=[o.get("lgd_std") or 0.0 for o in obligors])
    return {
        "method": "multi-factor-adjustment", "n_obligors": len(obligors), "n_factors": nf,
        "confidence_level": q, "expected_loss_rate": r.expected_loss, "var_rate": r.var,
        "expected_shortfall_rate": r.es, "var_limit_rate": r.var_limit,
        "expected_shortfall_limit_rate": r.es_limit,
    }


def _lists(x: Any) -> Any:
    """NumPy arrays and other sequences to plain lists (None passes through)."""
    if x is None:
        return None
    return x.tolist() if hasattr(x, "tolist") else [_lists(v) if hasattr(v, "tolist") else v for v in x]
