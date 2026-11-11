# cpm-approx

Analytical **Value at Risk** and **Expected Shortfall** of a credit portfolio in the multi-factor Merton model —
no simulation, deterministic, and fast enough for portfolios with thousands of facilities.

Monte Carlo is the usual tool for portfolio loss quantiles, but it is slow at high confidence levels and its result
changes with every seed. `cpm-approx` computes the quantile and the expected shortfall directly with the
*multi-factor adjustment* (M. Pykhtin, "Multi-factor adjustment", *Risk*, March 2004): the portfolio is replaced by a
comparable one-factor portfolio, and the effect of several systematic factors and of finite granularity is added as a
second-order correction. The result is a pair of closed-form expressions plus pairwise bivariate normal probabilities,
evaluated in `O(M²)` time and `O(M)` memory, in parallel.

This is an independent implementation of the published method; it is not affiliated with the author or the publisher.

- All figures are **loss rates** relative to total exposure; the loss quantile **includes the expected loss**.
- Stochastic LGD (mean and standard deviation per facility), any number of factors, any factor correlation matrix.
- Library crate `cpm_approx` and a JSON command line tool `cpm-approx`.

## Usage

```rust
use cpm_approx::{portfolio_risk, Portfolio};

let m = 100;
let p = Portfolio::from_weights(
    vec![0.02; m], vec![0.45; m], vec![1.0; m], vec![0.2; m],  // PD, LGD, exposure, R² (asset correlation)
    &vec![vec![1.0, 0.5]; m],                                  // sector weights per facility
    Some(&[vec![1.0, 0.3], vec![0.3, 1.0]]),                   // factor correlation matrix
    vec![0.0; m],                                              // LGD standard deviation
).unwrap();
let r = portfolio_risk(&p, 0.999).unwrap();
println!("VaR {:.4}  ES {:.4}  (one-factor limit {:.4})", r.var, r.es, r.var_limit);
```

Command line: JSON request on stdin (or a file path), JSON response on stdout.

```bash
cargo build --release
echo '{"confidence_level":0.999,"obligors":[{"pd":0.01,"lgd":0.45,"exposure":1,"asset_correlation":0.2}]}' \
  | RAYON_NUM_THREADS=2 target/release/cpm-approx
```

Request: `confidence_level`, optional `factor_correlation` (matrix; default: one factor), `obligors` with `pd`, `lgd`,
`exposure`, `asset_correlation` (R²), optional `factor_weights` and `lgd_std`. Response: `expected_loss_rate`,
`var_rate`, `expected_shortfall_rate`, and `var_limit_rate` / `expected_shortfall_limit_rate` (the one-factor
comparable portfolio without correction).

## Theory

### Model

A portfolio has `M` facilities with weights $w_i = \mathrm{EAD}_i / \sum_j \mathrm{EAD}_j$, default probability
$\mathrm{PD}_i$, mean loss given default $\mu_i$ and LGD standard deviation $\sigma_i$ (independent of everything
else). The asset value of facility $i$ depends on $K$ independent standard normal systematic factors $\eta$ and an
idiosyncratic factor $\xi_i$:

$$A_i = r_i\,\beta_i^\top \eta + \sqrt{1-r_i^2}\,\xi_i, \qquad \lVert\beta_i\rVert = 1,$$

and the facility defaults if $A_i \le \Phi^{-1}(\mathrm{PD}_i)$. Here $r_i^2$ is the asset correlation and
$\beta_i$ the loading on the factors (`Portfolio::from_weights` derives $\beta_i$ from sector weights and a factor
correlation matrix via its Cholesky factor). The portfolio loss rate is

$$L = \sum_i w_i\, \widetilde{\mathrm{LGD}}_i\, \mathbf 1\{A_i \le \Phi^{-1}(\mathrm{PD}_i)\}, \qquad
\mathbb E\,\widetilde{\mathrm{LGD}}_i=\mu_i .$$

We want the $q$-quantile $t_q$ (VaR) and $\mathrm{ES}_q = \mathbb E[L \mid L \ge t_q]$.

### Step 1: comparable one-factor portfolio

Collapse the factors to the single direction that matters most for the tail. With

$$c_i = w_i \mu_i\, \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}_i) + r_i\,\Phi^{-1}(q)}{\sqrt{1-r_i^2}}\right),
\qquad b = \frac{\sum_i c_i \beta_i}{\lVert\sum_i c_i \beta_i\rVert},$$

the factor $Y = b^\top\eta$ is standard normal and each asset has the **effective loading** $a_i = r_i\,\beta_i^\top b$
on it. Given $Y=y$ the conditional default probability and conditional expected loss are

$$p_i(y) = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}_i) - a_i y}{\sqrt{1-a_i^2}}\right), \qquad
\ell(y) = \sum_i w_i \mu_i\, p_i(y).$$

For a portfolio with infinitely many small facilities the loss equals $\ell(Y)$ and, since $\ell$ is decreasing,

$$t_0 = \ell(y_q), \quad y_q = \Phi^{-1}(1-q), \qquad
\mathrm{ES}_0 = \frac{1}{1-q}\sum_i w_i \mu_i\,\Phi_2\!\big(\Phi^{-1}(\mathrm{PD}_i),\, y_q;\, a_i\big),$$

where $\Phi_2(h,k;\rho)$ is the bivariate standard normal CDF. These are the `var_limit` / `es_limit` outputs.

### Step 2: second-order correction

The real loss is $L = \ell(Y) + \varepsilon$ with a residual $\varepsilon$ that has mean zero given $Y$ and conditional
variance $v(y)$. A second-order Taylor expansion of the quantile in $v$ gives

$$\Delta t = -\frac{1}{2\,\ell'(y_q)}\left[v'(y_q) - v(y_q)\left(\frac{\ell''(y_q)}{\ell'(y_q)} + y_q\right)\right],
\qquad t_q \approx t_0 + \Delta t,$$

and the corresponding correction for the expected shortfall is

$$\mathrm{ES}_q \approx \mathrm{ES}_0 - \frac{\varphi(y_q)}{2\,(1-q)}\,\frac{v(y_q)}{\ell'(y_q)} .$$

The conditional variance has two parts, $v = v_\infty + v_{GA}$. With $h_i = \Phi^{-1}(p_i(y))$ and the conditional
asset correlation

$$\rho_{ij} = \frac{r_i r_j\,\beta_i^\top\beta_j - a_i a_j}{\sqrt{(1-a_i^2)(1-a_j^2)}},$$

$$v_\infty = \sum_{i,j} w_i w_j \mu_i \mu_j\,\big[\Phi_2(h_i,h_j;\rho_{ij}) - p_i p_j\big],$$

$$v_{GA} = \sum_i w_i^2\,\Big[\mu_i^2\big(p_i - \Phi_2(h_i,h_i;\rho_{ii})\big) + \sigma_i^2\,p_i\Big].$$

$v_\infty$ is the **multi-factor adjustment**: the part of the variance that remains even for an infinitely granular
portfolio because the factors beyond $Y$ still move the loss. $v_{GA}$ is the **granularity adjustment** for the
finite number of facilities and for LGD risk. Their derivatives $v'$ are closed-form as well, for example

$$v'_\infty = 2\sum_{i,j} w_i w_j \mu_i\mu_j\, p_i'(y)\left[\Phi\!\left(\frac{h_j-\rho_{ij}h_i}{\sqrt{1-\rho_{ij}^2}}\right) - p_j\right].$$

`var_rate` is $t_0 + \Delta t$ with $v = v_\infty + v_{GA}$; the intermediate result with $v_\infty$ only is available
as `var_inf` in the library.

### Numerics

The bivariate normal CDF is evaluated with Plackett's identity,

$$\Phi_2(h,k;\rho) = \Phi(h)\Phi(k) + \int_0^{\rho} \frac{1}{2\pi\sqrt{1-r^2}}
\exp\!\left(-\frac{h^2 - 2rhk + k^2}{2(1-r^2)}\right) dr ,$$

by Gauss–Legendre quadrature with 5 to 24 nodes chosen by $|\rho|$ (error below $2\cdot10^{-15}$ against a 64-node
rule, tested). For $|\rho|\ge 0.9$ the substitution $r=\sin t$ with 40 nodes removes the endpoint singularity.
Pairs $(i,j)$ and $(j,i)$ are evaluated together, and the pair loop runs on the rayon thread pool.

### Scope and limits

The correction is a second-order expansion: it is accurate when the residual variance $v$ is small compared with the
spread of $\ell(Y)$, which holds for reasonably granular portfolios and gets worse for very few, very large
exposures. Validate against a simulation for unusual portfolios (strong concentration, extreme confidence levels
with few facilities). Euler risk contributions per facility are not implemented.

## Tests

`cargo test --release`

- `tests/table_c.rs` reproduces the published results of Table C in the paper (11 factors, 10 buckets, stochastic
  LGD, 99.9 %) to ±0.015 percentage points.
- `tests/parity.rs` compares against an independent Python reference implementation on synthetic random portfolios
  (1 to 200 facilities, 1 to 10 factors, stochastic LGD, confidence up to 99.98 %), relative tolerance $10^{-9}$.
- Unit tests for the quadrature, the bivariate normal and the single-factor closed form.

`cargo run --release --example bench -- 800 3200` prints the runtime on a synthetic portfolio. Indicative: two
cores, 10 factors, 800 facilities ≈ 0.1 s, 3200 facilities ≈ 1.4 s (cost grows quadratically).

## References

- M. Pykhtin, *Multi-factor adjustment*, Risk 17(3), 2004, 85–90.
- O. Vasicek, *The distribution of loan portfolio value*, Risk 15(12), 2002, 160–162.
- R. L. Plackett, *A reduction formula for normal multivariate integrals*, Biometrika 41, 1954, 351–360.

## License

MIT
