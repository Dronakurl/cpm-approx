//! Reproduction of the "Approx" columns of Table C in Pykhtin (2004), 11 factors (10 industries and
//! one global factor), 10 buckets, stochastic LGD, confidence 99.9 %.
use pykhtin::{pykhtin_risk, Portfolio};

const PD: [f64; 10] = [0.001, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05];
const MU: [f64; 10] = [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3];
const SIG: [f64; 10] = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1];
const R: [f64; 10] = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2];
const M_I: [usize; 10] = [50, 100, 50, 100, 50, 100, 50, 100, 50, 100];
const M_II: [usize; 10] = [10, 20, 10, 20, 10, 20, 10, 20, 10, 20];
const M_III: [usize; 10] = [10, 20, 50, 50, 100, 100, 200, 200, 500, 1000];

fn table_portfolio(counts: &[usize; 10], rho: f64) -> Portfolio {
    let alpha = rho.sqrt();
    let mut p = Portfolio { pd: vec![], lgd: vec![], exposure: vec![], r2: vec![], beta: vec![], lgd_std: vec![] };
    for (u, &m) in counts.iter().enumerate() {
        let mut beta = vec![0.0; 11];
        beta[10] = alpha;
        beta[u] = (1.0 - alpha * alpha).sqrt();
        for _ in 0..m {
            p.pd.push(PD[u]);
            p.lgd.push(MU[u]);
            p.exposure.push(0.1 / m as f64);
            p.r2.push(R[u] * R[u]);
            p.beta.push(beta.clone());
            p.lgd_std.push(SIG[u]);
        }
    }
    p
}

// (rho, counts, VaR %, ES %)
fn cases() -> Vec<(f64, [usize; 10], f64, f64)> {
    vec![
        (0.5, M_I, 2.33, 2.76),
        (0.5, M_II, 3.06, 3.55),
        (0.5, M_III, 2.32, 2.77),
        (0.3, M_I, 1.90, 2.18),
        (0.3, M_II, 2.80, 3.15),
        (0.1, M_I, 1.55, 1.71),
        (0.1, M_II, 2.82, 3.09),
    ]
}

#[test]
fn reproduces_published_table() {
    for (rho, counts, var, es) in cases() {
        let res = pykhtin_risk(&table_portfolio(&counts, rho), 0.999).unwrap();
        assert!((res.expected_loss - 0.00451).abs() < 2e-5);
        assert!((100.0 * res.var - var).abs() < 0.015, "rho {rho}: var {} vs {var}", 100.0 * res.var);
        assert!((100.0 * res.es - es).abs() < 0.015, "rho {rho}: es {} vs {es}", 100.0 * res.es);
    }
}

#[test]
fn limiting_loss_matches_published_value() {
    let res = pykhtin_risk(&table_portfolio(&M_I, 0.5), 0.999).unwrap();
    assert!((100.0 * res.var_inf - 2.15).abs() < 0.015);
}
