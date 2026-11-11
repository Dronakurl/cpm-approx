# cpm-approx

Analytical **Value at Risk** and **Expected Shortfall** of a credit portfolio in the multi-factor Merton model — no
simulation, deterministic, fast for portfolios with thousands of facilities. It implements the *multi-factor adjustment*
of M. Pykhtin, "Multi-factor adjustment", *Risk*, March 2004, in Rust with Python bindings. It is not affiliated with the
author of the paper or its publisher.

All figures are **loss rates** relative to total exposure; the loss quantile **includes the expected loss**.

## Install

Python 3.12 or newer, no dependencies (NumPy arrays are accepted as input).

```bash
pip install cpm-approx            # once released on PyPI
pip install --index-url https://test.pypi.org/simple/ cpm-approx    # current test release
```

## Usage

The example portfolio is the one of the paper (Table C: 11 factors, ten buckets, 750 facilities):

```python
from cpm_approx import paper_portfolio, portfolio_risk

r = portfolio_risk(**paper_portfolio(rho=0.5), confidence=0.999)
print(f"VaR {r.var:.2%}  ES {r.es:.2%}  one-factor limit {r.var_limit:.2%}")    # VaR 2.33%  ES 2.76%
```

Your own portfolio, one entry per facility:

```python
r = portfolio_risk(
    pd, lgd, exposure, asset_correlation, 0.999,
    factor_weights=weights,            # sector weights per facility ...
    factor_correlation=corr,           # ... and the correlation matrix of the factors
    lgd_std=lgd_std,                   # optional: standard deviation of the LGD
)
```

- `asset_correlation` is the squared loading R² on the systematic factors.
- Instead of weights and correlation matrix you can pass `loadings` (rows of unit norm on independent factors).
  Without any factor input a single factor is used.
- The result `RiskResult` has `expected_loss`, `var`, `es` and, for comparison, `var_limit` and `es_limit` of the
  comparable one-factor portfolio before the multi-factor and granularity corrections.
- `analyze(dict)` takes and returns JSON-like dictionaries. Invalid input raises `ValueError`.
  The GIL is released during the computation.

The cost grows quadratically with the number of facilities (3000 facilities ≈ 1 s on two cores). The method is a
second-order expansion; for very concentrated portfolios with few facilities, check against a simulation.

## Development

```bash
uv sync && uv run pytest      # builds the Rust extension (Rust toolchain required)
cargo test --release          # Rust tests: Table C of the paper, parity with a Python reference implementation
```

The Rust library (`cpm_approx`) and a JSON command line tool (`cpm-approx`) are built from the same sources.

## Reference

M. Pykhtin, *Multi-factor adjustment*, Risk 17(3), 2004, 85–90.

## License

MIT
