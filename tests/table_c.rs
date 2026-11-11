//! Reproduction of the "Approx" columns of Table C in Pykhtin (2004), 11 factors (10 industries and
//! one global factor), 10 buckets, stochastic LGD, confidence 99.9 %.
use cpm_approx::paper::{table_c_portfolio, COUNTS_I, COUNTS_II, COUNTS_III};
use cpm_approx::portfolio_risk;

// (rho, counts, VaR %, ES %)
fn cases() -> Vec<(f64, [usize; 10], f64, f64)> {
    vec![
        (0.5, COUNTS_I, 2.33, 2.76),
        (0.5, COUNTS_II, 3.06, 3.55),
        (0.5, COUNTS_III, 2.32, 2.77),
        (0.3, COUNTS_I, 1.90, 2.18),
        (0.3, COUNTS_II, 2.80, 3.15),
        (0.1, COUNTS_I, 1.55, 1.71),
        (0.1, COUNTS_II, 2.82, 3.09),
    ]
}

#[test]
fn reproduces_published_table() {
    for (rho, counts, var, es) in cases() {
        let res = portfolio_risk(&table_c_portfolio(&counts, rho), 0.999).unwrap();
        assert!((res.expected_loss - 0.00451).abs() < 2e-5);
        assert!((100.0 * res.var - var).abs() < 0.015, "rho {rho}: var {} vs {var}", 100.0 * res.var);
        assert!((100.0 * res.es - es).abs() < 0.015, "rho {rho}: es {} vs {es}", 100.0 * res.es);
    }
}

#[test]
fn limiting_loss_matches_published_value() {
    let res = portfolio_risk(&table_c_portfolio(&COUNTS_I, 0.5), 0.999).unwrap();
    assert!((100.0 * res.var_inf - 2.15).abs() < 0.015);
}
