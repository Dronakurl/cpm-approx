//! Python bindings: thin wrappers around `cpm_approx`. The public Python API lives in
//! `pysrc/cpm_approx/__init__.py`; this module is `cpm_approx._native`.
use cpm_approx::{paper, portfolio_risk, Portfolio};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Risk = (f64, f64, f64, f64, f64, f64);

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// (expected_loss, var_limit, var_inf, var, es_limit, es)
#[pyfunction]
#[pyo3(signature = (pd, lgd, exposure, asset_correlation, confidence, factor_weights=None, factor_correlation=None, loadings=None, lgd_std=None))]
#[allow(clippy::too_many_arguments)]
fn risk(
    py: Python<'_>,
    pd: Vec<f64>,
    lgd: Vec<f64>,
    exposure: Vec<f64>,
    asset_correlation: Vec<f64>,
    confidence: f64,
    factor_weights: Option<Vec<Vec<f64>>>,
    factor_correlation: Option<Vec<Vec<f64>>>,
    loadings: Option<Vec<Vec<f64>>>,
    lgd_std: Option<Vec<f64>>,
) -> PyResult<Risk> {
    let m = pd.len();
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(err("confidence must lie in (0, 1)".into()));
    }
    if [lgd.len(), exposure.len(), asset_correlation.len()].iter().any(|&l| l != m) {
        return Err(err("pd, lgd, exposure and asset_correlation must have the same length".into()));
    }
    let lgd_std = lgd_std.unwrap_or_else(|| vec![0.0; m]);
    let result = py.detach(move || -> Result<Risk, String> {
        let portfolio = match (factor_weights, loadings) {
            (Some(_), Some(_)) => return Err("give either factor_weights or loadings, not both".into()),
            (_, Some(beta)) => {
                let p = Portfolio { pd, lgd, exposure, r2: asset_correlation, beta, lgd_std };
                p.validate()?;
                p
            }
            (weights, None) => {
                let weights = match weights {
                    Some(w) => {
                        if w.len() != m {
                            return Err("factor_weights must have one row per facility".into());
                        }
                        let nf = factor_correlation.as_ref().map_or_else(|| w.iter().map(Vec::len).max().unwrap_or(1), Vec::len);
                        w.into_iter()
                            .map(|mut row| {
                                row.resize(nf.max(row.len()), 0.0);
                                row
                            })
                            .collect::<Vec<_>>()
                    }
                    None => {
                        let nf = factor_correlation.as_ref().map_or(1, Vec::len);
                        (0..m).map(|_| { let mut r = vec![0.0; nf]; r[0] = 1.0; r }).collect()
                    }
                };
                if let Some(c) = &factor_correlation {
                    if weights.iter().any(|r| r.len() != c.len()) {
                        return Err("factor_weights has more columns than factor_correlation".into());
                    }
                }
                Portfolio::from_weights(pd, lgd, exposure, asset_correlation, &weights, factor_correlation.as_deref(), lgd_std)?
            }
        };
        let r = portfolio_risk(&portfolio, confidence)?;
        Ok((r.expected_loss, r.var_limit, r.var_inf, r.var, r.es_limit, r.es))
    });
    result.map_err(err)
}

type PaperPortfolio = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

/// (pd, lgd, exposure, asset_correlation, lgd_std, loadings) of the paper's example portfolio
#[pyfunction]
fn paper_portfolio(rho: f64, counts: Vec<usize>) -> PyResult<PaperPortfolio> {
    let counts: [usize; 10] = counts.try_into().map_err(|_| err("counts must have 10 entries".into()))?;
    if !(0.0..1.0).contains(&rho) {
        return Err(err("rho must lie in [0, 1)".into()));
    }
    let p = paper::table_c_portfolio(&counts, rho);
    Ok((p.pd, p.lgd, p.exposure, p.r2, p.lgd_std, p.beta))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(risk, m)?)?;
    m.add_function(wrap_pyfunction!(paper_portfolio, m)?)?;
    m.add("COUNTS_I", paper::COUNTS_I.to_vec())?;
    m.add("COUNTS_II", paper::COUNTS_II.to_vec())?;
    m.add("COUNTS_III", paper::COUNTS_III.to_vec())?;
    Ok(())
}
