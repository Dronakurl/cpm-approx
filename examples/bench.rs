//! Runtime on a synthetic portfolio: `cargo run --release --example bench -- 800 3200`
//! Facilities are spread over 10 factors with a constant factor correlation of 0.3.
use std::time::Instant;

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let sizes = if sizes.is_empty() { vec![200, 800, 3200] } else { sizes };
    let nf = 10;
    let corr: Vec<Vec<f64>> = (0..nf)
        .map(|i| (0..nf).map(|j| if i == j { 1.0 } else { 0.3 }).collect())
        .collect();
    for m in sizes {
        let obligors = (0..m)
            .map(|i| {
                let mut w = vec![0.0; nf];
                w[i % nf] = 1.0;
                cpm_approx::Obligor {
                    pd: 0.002 + 0.03 * ((i * 7919) % 100) as f64 / 100.0,
                    lgd: 0.2 + 0.4 * ((i * 104_729) % 100) as f64 / 100.0,
                    exposure: 1.0 + ((i * 31) % 50) as f64,
                    asset_correlation: 0.15,
                    factor_weights: Some(w),
                    lgd_std: None,
                }
            })
            .collect();
        let req = cpm_approx::Request { confidence_level: 0.999, factor_correlation: Some(corr.clone()), obligors };
        let t = Instant::now();
        let r = cpm_approx::analyze(&req).expect("analysis");
        println!("M={m:6}  {:7.3} s  VaR {:.5}  ES {:.5}", t.elapsed().as_secs_f64(), r.var_rate, r.expected_shortfall_rate);
    }
}
