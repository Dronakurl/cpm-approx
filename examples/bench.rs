//! Runtime on the example portfolio of the paper (Table C, 11 factors), scaled to more facilities:
//! `cargo run --release --example bench -- 1 4 13` (multiples of the 750 facilities of portfolio I).
use cpm_approx::paper::{table_c_portfolio, COUNTS_I};
use cpm_approx::portfolio_risk;
use std::time::Instant;

fn main() {
    let scales: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let scales = if scales.is_empty() { vec![1, 4] } else { scales };
    for k in scales {
        let counts = COUNTS_I.map(|c| c * k);
        let p = table_c_portfolio(&counts, 0.3);
        let t = Instant::now();
        let r = portfolio_risk(&p, 0.999).expect("analysis");
        println!(
            "M={:6}  {:7.3} s  VaR {:.5}  ES {:.5}",
            p.pd.len(),
            t.elapsed().as_secs_f64(),
            r.var,
            r.es
        );
    }
}
