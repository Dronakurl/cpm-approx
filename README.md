# pykhtin-rs

Analytical Value at Risk and Expected Shortfall of a credit portfolio in the multi-factor Merton model,
using the multi-factor adjustment of

> M. Pykhtin, *Multi-factor adjustment*, Risk, March 2004, pp. 85-90.

The portfolio is replaced by a comparable one-factor portfolio; the multi-factor structure and the finite
granularity enter as a second-order Taylor correction of the loss quantile, and the expected shortfall follows in
closed form with bivariate normal probabilities. The cost is O(M²) in the number of facilities and runs in
parallel (rayon). Memory is O(M).

All figures are loss rates relative to total exposure; the loss quantile includes the expected loss.

## Usage

```rust
use pykhtin::{Portfolio, pykhtin_risk};

let m = 100;
let p = Portfolio::from_weights(
    vec![0.02; m], vec![0.45; m], vec![1.0; m], vec![0.2; m],
    &vec![vec![1.0, 0.5]; m],                  // sector weights per facility
    Some(&[vec![1.0, 0.3], vec![0.3, 1.0]]),   // factor correlation matrix
    vec![0.0; m],                              // LGD standard deviation
).unwrap();
let r = pykhtin_risk(&p, 0.999).unwrap();
println!("VaR {:.4}  ES {:.4}  (one-factor limit {:.4})", r.var, r.es, r.var_limit);
```

Command line: the binary reads a JSON request from stdin (or a file path) and prints a JSON response.

```bash
cargo build --release
echo '{"confidence_level":0.999,"obligors":[{"pd":0.01,"lgd":0.45,"exposure":1,"asset_correlation":0.2}]}' \
  | RAYON_NUM_THREADS=2 target/release/pykhtin-rs
```

Request fields: `confidence_level`, optional `factor_correlation` (matrix; default one factor), `obligors` with
`pd`, `lgd`, `exposure`, `asset_correlation` (R²), optional `factor_weights` and `lgd_std`.
Response: `expected_loss_rate`, `var_rate`, `expected_shortfall_rate`, `var_limit_rate` and
`expected_shortfall_limit_rate` (the one-factor comparable portfolio without correction).

## Model

`asset_i = sqrt(R²_i) · Σ_k β_ik η_k + sqrt(1 − R²_i) · ξ_i`, default iff `asset_i ≤ Φ⁻¹(PD_i)`, independent standard
normal factors `η`, loadings `β_i` of unit norm (`Portfolio::from_weights` derives them from sector weights and a factor
correlation matrix). Stochastic LGD with mean `lgd` and standard deviation `lgd_std`, independent of everything else.
Not included: Euler risk contributions per facility.

## Implementation notes

- The bivariate normal CDF uses Plackett's identity `N₂(h,k;ρ) = Φ(h)Φ(k) + ∫₀^ρ exp(−(h²−2rhk+k²)/(2(1−r²))) / (2π√(1−r²)) dr`
  with Gauss-Legendre quadrature of 5 to 24 nodes chosen by |ρ| (error below 2e-15 against a 64-node rule, tested).
  For |ρ| ≥ 0.9 the substitution `r = sin t` with 40 nodes avoids the endpoint singularity.
- Pairs (i, j) and (j, i) are evaluated together, known marginals are reused.

## Tests

`cargo test --release`

- `tests/table_c.rs`: reproduces the published "Approx" results of Table C in Pykhtin (2004) to ±0.015 percentage points.
- `tests/parity.rs`: matches the Python reference implementation on synthetic random portfolios
  (1 to 200 facilities, 1 to 10 factors, stochastic LGD, confidence up to 99.98 %), relative tolerance 1e-9.
- Unit tests for the quadrature, the bivariate normal and the single-factor closed form.

`cargo run --release --example bench -- 800 3200` prints the runtime on a synthetic portfolio.
Indicative: on two cores a 10-factor portfolio with 800 facilities takes about 0.1 s and 3200 facilities about 1.4 s (synthetic portfolio, cost grows quadratically).

## License

MIT
