//! Analytical VaR and expected shortfall of a credit portfolio in the multi-factor Merton model
//! (multi-factor adjustment of Pykhtin, Risk, March 2004).
//!
//! Rust implementation (same formulas and conventions as the Python reference implementation): loss rates
//! relative to total exposure, loss quantile including the expected loss, cost O(M^2) in the
//! number of facilities (pairwise bivariate normals), parallelised over facilities.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::erf::erfc;
use std::f64::consts::{PI, SQRT_2};
use std::sync::OnceLock;

const RHO_CLIP: f64 = 0.999999;

/// Gauss-Legendre rule sizes for the Plackett integral in the variable `r`, chosen by `|rho|`:
/// `(upper limit of |rho|, nodes)`. Measured against a 64-node rule over |h|, |k| <= 8 the error
/// of every tier is below 1e-15. For `|rho| >= 0.9` the integrand has a near-singularity at the end
/// of the interval; there the substitution `r = sin t` with the 40 nodes of the Python library is used.
const TIERS: [(f64, usize); 6] = [(0.1, 5), (0.2, 6), (0.4, 8), (0.5, 10), (0.7, 16), (0.9, 24)];
const FULL_NODES: usize = 40;

/// Standard normal CDF
#[inline]
pub fn cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Standard normal density
#[inline]
pub fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal quantile function
pub fn ppf(p: f64) -> f64 {
    static N: OnceLock<Normal> = OnceLock::new();
    N.get_or_init(|| Normal::new(0.0, 1.0).expect("standard normal"))
        .inverse_cdf(p)
}

/// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on the Legendre polynomial)
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp;
        loop {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

type Rule = (Vec<f64>, Vec<f64>);

fn rules() -> &'static [Rule; 7] {
    static R: OnceLock<[Rule; 7]> = OnceLock::new();
    R.get_or_init(|| {
        let mut v: Vec<Rule> = TIERS.iter().map(|&(_, n)| gauss_legendre(n)).collect();
        v.push(gauss_legendre(FULL_NODES));
        v.try_into().expect("seven rules")
    })
}

/// Bivariate standard normal CDF N2(h, k; rho).
///
/// Plackett identity `N2 = Phi(h)Phi(k) + int_0^rho exp(-(h^2 - 2 r h k + k^2) / (2 (1 - r^2))) / (2 pi sqrt(1 - r^2)) dr`
/// by Gauss-Legendre quadrature.
pub fn n2(h: f64, k: f64, rho: f64) -> f64 {
    n2_phi(cdf(h), cdf(k), h, k, rho)
}

/// `n2` with the marginal probabilities `phi_h = Phi(h)`, `phi_k = Phi(k)` already known
#[inline]
pub fn n2_phi(phi_h: f64, phi_k: f64, h: f64, k: f64, rho: f64) -> f64 {
    let rho = rho.clamp(-1.0 + 1e-12, 1.0 - 1e-12);
    phi_h * phi_k + plackett(h, k, rho)
}

fn plackett(h: f64, k: f64, rho: f64) -> f64 {
    let a = rho.abs();
    let Some(rule) = TIERS.iter().position(|&(lim, _)| a < lim) else {
        return plackett_sine(h, k, rho);
    };
    let (xs, ws) = &rules()[rule];
    let half = rho / 2.0;
    let (hh, hk2) = (h * h + k * k, 2.0 * h * k);
    let mut out = 0.0;
    for (x, w) in xs.iter().zip(ws) {
        let r = half * (x + 1.0);
        let om = 1.0 - r * r;
        let inv = 1.0 / om;
        out += w * (-0.5 * (hh - hk2 * r) * inv).exp() * inv.sqrt();
    }
    out * half / (2.0 * PI)
}

/// Variant with `r = sin t` for strongly correlated pairs (no endpoint singularity)
fn plackett_sine(h: f64, k: f64, rho: f64) -> f64 {
    let (xs, ws) = &rules()[TIERS.len()];
    let half = rho.asin() / 2.0;
    let (hh, hk2) = (h * h + k * k, 2.0 * h * k);
    let mut out = 0.0;
    for (x, w) in xs.iter().zip(ws) {
        let (s, c) = (half * (x + 1.0)).sin_cos();
        out += w * (-(hh - hk2 * s) / (2.0 * c * c)).exp();
    }
    out * half / (2.0 * PI)
}

/// Portfolio of the multi-factor Merton model (default mode, loss rates).
///
/// `asset_i = sqrt(r2_i) sum_k beta_ik eta_k + sqrt(1 - r2_i) xi_i`, default iff `asset_i <= N^-1(pd_i)`.
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub pd: Vec<f64>,
    pub lgd: Vec<f64>,
    pub exposure: Vec<f64>,
    pub r2: Vec<f64>,
    /// Loadings on independent standard normal factors, one unit-norm row per facility
    pub beta: Vec<Vec<f64>>,
    pub lgd_std: Vec<f64>,
}

impl Portfolio {
    pub fn validate(&self) -> Result<(), String> {
        let m = self.pd.len();
        if m == 0 {
            return Err("obligors must not be empty".into());
        }
        if [self.lgd.len(), self.exposure.len(), self.r2.len(), self.lgd_std.len(), self.beta.len()]
            .iter()
            .any(|&l| l != m)
        {
            return Err("all columns must have one entry per facility".into());
        }
        for i in 0..m {
            if !(self.pd[i] > 0.0 && self.pd[i] < 1.0) {
                return Err("pd must lie strictly in (0, 1)".into());
            }
            if !(self.r2[i] > 0.0 && self.r2[i] < 1.0) {
                return Err("r2 must lie strictly in (0, 1)".into());
            }
            if self.exposure[i].is_nan() || self.exposure[i] <= 0.0 {
                return Err("exposure must be positive".into());
            }
            let n: f64 = self.beta[i].iter().map(|b| b * b).sum::<f64>().sqrt();
            if (n - 1.0).abs() > 1e-9 {
                return Err("rows of beta must have unit norm".into());
            }
        }
        Ok(())
    }

    /// Loadings from sector weights and the factor correlation matrix (convention: weights are normalised, factors correlated through the Cholesky factor of `corr`):
    /// `w~ = w/|w|`, `X = L z` with `corr = L L^T`, `beta_i = L^T w~ / |L^T w~|`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        pd: Vec<f64>,
        lgd: Vec<f64>,
        exposure: Vec<f64>,
        r2: Vec<f64>,
        weights: &[Vec<f64>],
        corr: Option<&[Vec<f64>]>,
        lgd_std: Vec<f64>,
    ) -> Result<Self, String> {
        let l = match corr {
            Some(c) => Some(cholesky(c)?),
            None => None,
        };
        let mut beta = Vec::with_capacity(weights.len());
        for w in weights {
            let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n.is_nan() || n <= 0.0 {
                return Err("factor weights are all zero".into());
            }
            let mut v: Vec<f64> = w.iter().map(|x| x / n).collect();
            if let Some(l) = &l {
                let nf = v.len();
                v = (0..nf).map(|k| (k..nf).map(|j| v[j] * l[j][k]).sum()).collect();
            }
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            beta.push(v.iter().map(|x| x / n).collect());
        }
        let p = Portfolio { pd, lgd, exposure, r2, beta, lgd_std };
        p.validate()?;
        Ok(p)
    }

    pub fn n_factors(&self) -> usize {
        self.beta.first().map_or(0, Vec::len)
    }
}

/// Lower Cholesky factor
pub fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        if a[i].len() != n {
            return Err("factor_correlation must be square".into());
        }
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d <= 0.0 {
                    return Err("factor_correlation is not positive definite".into());
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Ok(l)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RiskResult {
    pub expected_loss: f64,
    /// Comparable one-factor portfolio
    pub var_limit: f64,
    /// Plus systematic multi-factor adjustment (M -> infinity)
    pub var_inf: f64,
    /// Plus granularity adjustment
    pub var: f64,
    pub es_limit: f64,
    pub es: f64,
}

/// Effective loadings `a_i = r_i sum_k beta_ik b_k` (paper eqs. 20-21), `r_i`, weights `w_i`
fn effective_loadings(p: &Portfolio, q: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let total: f64 = p.exposure.iter().sum();
    let w: Vec<f64> = p.exposure.iter().map(|e| e / total).collect();
    let r: Vec<f64> = p.r2.iter().map(|x| x.sqrt()).collect();
    let nf = p.n_factors();
    let zq = ppf(q);
    let mut bvec = vec![0.0; nf];
    for i in 0..p.pd.len() {
        let c = w[i] * p.lgd[i] * cdf((ppf(p.pd[i]) + r[i] * zq) / (1.0 - r[i] * r[i]).sqrt());
        for (b, x) in bvec.iter_mut().zip(&p.beta[i]) {
            *b += c * x;
        }
    }
    let nb = bvec.iter().map(|x| x * x).sum::<f64>().sqrt();
    let a = (0..p.pd.len())
        .map(|i| r[i] * (0..nf).map(|k| p.beta[i][k] * bvec[k] / nb).sum::<f64>())
        .collect();
    (a, r, w)
}

/// Multi-factor-adjustment VaR and ES at confidence `q` (loss rates). Runs on the rayon thread pool.
pub fn portfolio_risk(p: &Portfolio, q: f64) -> Result<RiskResult, String> {
    p.validate()?;
    if !(q > 0.0 && q < 1.0) {
        return Err("confidence level must lie in (0, 1)".into());
    }
    let m = p.pd.len();
    let nf = p.n_factors();
    let sig = &p.lgd_std;
    let (a, r, w) = effective_loadings(p, q);
    let mu = &p.lgd;
    let y = ppf(1.0 - q);
    let cpd: Vec<f64> = p.pd.iter().map(|&x| ppf(x)).collect();
    let s: Vec<f64> = a.iter().map(|x| (1.0 - x * x).sqrt()).collect();

    // conditional PD and derivatives at y (eqs. 9, 23, 24)
    let z: Vec<f64> = (0..m).map(|i| (cpd[i] - a[i] * y) / s[i]).collect();
    let ph: Vec<f64> = z.iter().map(|&x| cdf(x)).collect();
    let nz: Vec<f64> = z.iter().map(|&x| pdf(x)).collect();
    let ph1: Vec<f64> = (0..m).map(|i| -a[i] / s[i] * nz[i]).collect();
    let ph2: Vec<f64> = (0..m).map(|i| -(a[i] * a[i]) / (1.0 - a[i] * a[i]) * z[i] * nz[i]).collect();
    let l: f64 = (0..m).map(|i| w[i] * mu[i] * ph[i]).sum();
    let l1: f64 = (0..m).map(|i| w[i] * mu[i] * ph1[i]).sum();
    let l2: f64 = (0..m).map(|i| w[i] * mu[i] * ph2[i]).sum();
    let el: f64 = (0..m).map(|i| w[i] * mu[i] * p.pd[i]).sum();

    // pairwise part (eq. 26): rho^Y_ij = (G_ij - a_i a_j)/(s_i s_j), G_ij = r_i r_j sum_k beta_ik beta_jk
    let rb: Vec<f64> = (0..m)
        .flat_map(|i| {
            let ri = r[i];
            p.beta[i].iter().map(move |b| ri * b)
        })
        .collect();
    let nzh: Vec<f64> = ph.iter().map(|&x| ppf(x.clamp(1e-300, 1.0 - 1e-16))).collect();
    let phh: Vec<f64> = nzh.iter().map(|&x| cdf(x)).collect();
    let wm: Vec<f64> = (0..m).map(|i| w[i] * mu[i]).collect();

    let (v_inf, v1_inf) = (0..m)
        .into_par_iter()
        .map(|i| {
            let (mut v, mut v1) = (0.0, 0.0);
            let ri = &rb[i * nf..(i + 1) * nf];
            for j in i..m {
                let g: f64 = ri.iter().zip(&rb[j * nf..(j + 1) * nf]).map(|(x, y)| x * y).sum();
                let rho = ((g - a[i] * a[j]) / (s[i] * s[j])).clamp(-RHO_CLIP, RHO_CLIP);
                let (h, k) = (nzh[i], nzh[j]);
                let sq = (1.0 - rho * rho).sqrt();
                let n2m = n2_phi(phh[i], phh[j], h, k, rho);
                let inner_ij = cdf((k - rho * h) / sq) - ph[j];
                let t = wm[i] * wm[j];
                if i == j {
                    v += t * (n2m - ph[i] * ph[i]);
                    v1 += 2.0 * t * ph1[i] * inner_ij;
                } else {
                    // the (j, i) term of the full double sum is identical for v, and swaps h/k for v1
                    let inner_ji = cdf((h - rho * k) / sq) - ph[i];
                    v += 2.0 * t * (n2m - ph[i] * ph[j]);
                    v1 += 2.0 * t * (ph1[i] * inner_ij + ph1[j] * inner_ji);
                }
            }
            (v, v1)
        })
        .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));

    // granularity part (eqs. 30, 31): diagonal i = j
    let (mut v_ga, mut v1_ga) = (0.0, 0.0);
    for i in 0..m {
        let rho = ((r[i] * r[i] - a[i] * a[i]) / (1.0 - a[i] * a[i])).clamp(-RHO_CLIP, RHO_CLIP);
        let w2 = w[i] * w[i];
        let n2ii = n2(nzh[i], nzh[i], rho);
        v_ga += w2 * (mu[i] * mu[i] * (ph[i] - n2ii) + sig[i] * sig[i] * ph[i]);
        let c = cdf((nzh[i] - rho * nzh[i]) / (1.0 - rho * rho).sqrt());
        v1_ga += w2 * ph1[i] * (mu[i] * mu[i] * (1.0 - 2.0 * c) + sig[i] * sig[i]);
    }

    // eq. (22)
    let delta_t = |v: f64, v1: f64| -1.0 / (2.0 * l1) * (v1 - v * (l2 / l1 + y));
    let d_inf = delta_t(v_inf, v1_inf);
    let d_ga = delta_t(v_ga, v1_ga);

    // expected shortfall, eqs. (36)-(38)
    let es_l: f64 = (0..m).map(|i| w[i] * mu[i] * n2(cpd[i], y, a[i])).sum::<f64>() / (1.0 - q);
    let d_es = |v: f64| -1.0 / (2.0 * (1.0 - q)) * pdf(y) * v / l1;
    Ok(RiskResult {
        expected_loss: el,
        var_limit: l,
        var_inf: l + d_inf,
        var: l + d_inf + d_ga,
        es_limit: es_l,
        es: es_l + d_es(v_inf) + d_es(v_ga),
    })
}

// ---------------------------------------------------------------- JSON entry point

#[derive(Debug, Deserialize)]
pub struct Obligor {
    pub pd: f64,
    pub lgd: f64,
    pub exposure: f64,
    pub asset_correlation: f64,
    #[serde(default)]
    pub factor_weights: Option<Vec<f64>>,
    #[serde(default)]
    pub lgd_std: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Request {
    pub confidence_level: f64,
    #[serde(default)]
    pub factor_correlation: Option<Vec<Vec<f64>>>,
    pub obligors: Vec<Obligor>,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub method: &'static str,
    pub n_obligors: usize,
    pub n_factors: usize,
    pub confidence_level: f64,
    pub expected_loss_rate: f64,
    pub var_rate: f64,
    pub expected_shortfall_rate: f64,
    pub var_limit_rate: f64,
    pub expected_shortfall_limit_rate: f64,
}

pub fn portfolio_from_request(req: &Request) -> Result<Portfolio, String> {
    if req.obligors.is_empty() {
        return Err("obligors must not be empty".into());
    }
    let corr = req.factor_correlation.as_deref().filter(|c| !c.is_empty());
    let nf = corr.map_or(1, <[_]>::len);
    let mut weights = Vec::with_capacity(req.obligors.len());
    for (i, o) in req.obligors.iter().enumerate() {
        let mut row = vec![0.0; nf];
        match o.factor_weights.as_deref().filter(|w| !w.is_empty()) {
            Some(w) => {
                if w.len() > nf {
                    return Err(format!("obligor {i}: {} factor weights but {nf} factors", w.len()));
                }
                row[..w.len()].copy_from_slice(w);
            }
            None => row[0] = 1.0,
        }
        let norm2: f64 = row.iter().map(|x| x * x).sum();
        if norm2.is_nan() || norm2 <= 0.0 {
            return Err(format!("obligor {i}: factor weights are all zero"));
        }
        weights.push(row);
    }
    let o = &req.obligors;
    Portfolio::from_weights(
        o.iter().map(|x| x.pd).collect(),
        o.iter().map(|x| x.lgd).collect(),
        o.iter().map(|x| x.exposure).collect(),
        o.iter().map(|x| x.asset_correlation).collect(),
        &weights,
        corr,
        o.iter().map(|x| x.lgd_std.unwrap_or(0.0)).collect(),
    )
}

/// JSON-level entry point: request in, response out (no per-facility contributions)
pub fn analyze(req: &Request) -> Result<Response, String> {
    let q = req.confidence_level;
    if !(q > 0.0 && q < 1.0) {
        return Err("confidence_level must lie in (0, 1)".into());
    }
    let p = portfolio_from_request(req)?;
    let r = portfolio_risk(&p, q)?;
    Ok(Response {
        method: "multi-factor-adjustment",
        n_obligors: p.pd.len(),
        n_factors: p.n_factors(),
        confidence_level: q,
        expected_loss_rate: r.expected_loss,
        var_rate: r.var,
        expected_shortfall_rate: r.es,
        var_limit_rate: r.var_limit,
        expected_shortfall_limit_rate: r.es_limit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        let (x, w) = gauss_legendre(40);
        let i2: f64 = x.iter().zip(&w).map(|(x, w)| w * x * x).sum();
        assert!((i2 - 2.0 / 3.0).abs() < 1e-14);
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn n2_known_values() {
        assert!((n2(0.0, 0.0, 0.5) - (0.25 + 0.5_f64.asin() / (2.0 * PI))).abs() < 1e-13);
        assert!((n2(0.3, -0.7, 0.0) - cdf(0.3) * cdf(-0.7)).abs() < 1e-15);
        // perfect correlation -> min of the marginals
        assert!((n2(0.3, -0.7, 1.0) - cdf(-0.7)).abs() < 1e-6);
    }

    #[test]
    fn tiered_rules_match_reference_rule() {
        // reference: 64-node sine-substitution rule, valid for every rho
        let reference = |h: f64, k: f64, rho: f64| {
            let (xs, ws) = gauss_legendre(64);
            let half = rho.asin() / 2.0;
            xs.iter()
                .zip(&ws)
                .map(|(x, w)| {
                    let (s, c) = (half * (x + 1.0)).sin_cos();
                    half * w * (-(h * h - 2.0 * s * h * k + k * k) / (2.0 * c * c)).exp() / (2.0 * PI)
                })
                .sum::<f64>()
        };
        let mut worst: f64 = 0.0;
        for hi in -8..=8 {
            for ki in -8..=8 {
                for ri in -40..=40 {
                    let (h, k) = (f64::from(hi), f64::from(ki));
                    let rho = f64::from(ri) / 40.0 * 0.99;
                    let got = n2(h, k, rho) - cdf(h) * cdf(k);
                    worst = worst.max((got - reference(h, k, rho)).abs());
                }
            }
        }
        assert!(worst < 2e-15, "worst difference {worst:e}");
    }

    #[test]
    fn normal_inverse_roundtrip() {
        for p in [1e-6, 0.01, 0.3, 0.5, 0.95, 0.999] {
            let e = ((cdf(ppf(p)) - p) / p).abs(); assert!(e < 1e-9, "p={p} rel err {e}");
        }
    }

    #[test]
    fn single_factor_equals_closed_form() {
        // one factor, homogeneous: var_limit == Vasicek loss quantile, correction ~ 0 for huge M
        let m = 200;
        let p = Portfolio::from_weights(
            vec![0.02; m], vec![0.45; m], vec![1.0; m], vec![0.2; m],
            &vec![vec![1.0]; m], None, vec![0.0; m],
        ).unwrap();
        let res = portfolio_risk(&p, 0.999).unwrap();
        let vas = 0.45 * cdf((ppf(0.02) + 0.2_f64.sqrt() * ppf(0.999)) / 0.8_f64.sqrt());
        assert!((res.var_limit - vas).abs() < 1e-12);
        assert!(res.var > res.var_limit); // granularity adds
        assert!(res.es > res.var);
    }
}


