//! Parity with the Python reference implementation (`pykhtin`): synthetic random portfolios with
//! 1 to 200 facilities, 1 to 10 factors, stochastic LGD and confidence levels up to 99.98 %.
//! The expected values in `fixtures/parity.json` were produced by the Python library.
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    request: pykhtin::Request,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    expected_loss_rate: f64,
    var_rate: f64,
    expected_shortfall_rate: f64,
    var_limit_rate: f64,
    expected_shortfall_limit_rate: f64,
}

#[test]
fn matches_python_reference() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/parity.json")).expect("fixtures");
    assert!(cases.len() >= 7);
    for (i, c) in cases.iter().enumerate() {
        let got = pykhtin::analyze(&c.request).expect("analysis");
        let pairs = [
            ("expected_loss", got.expected_loss_rate, c.expected.expected_loss_rate),
            ("var", got.var_rate, c.expected.var_rate),
            ("es", got.expected_shortfall_rate, c.expected.expected_shortfall_rate),
            ("var_limit", got.var_limit_rate, c.expected.var_limit_rate),
            ("es_limit", got.expected_shortfall_limit_rate, c.expected.expected_shortfall_limit_rate),
        ];
        for (name, g, e) in pairs {
            assert!(
                (g - e).abs() <= 1e-9 * e.abs() + 1e-14,
                "case {i} {name}: rust {g} vs python {e}"
            );
        }
    }
}

#[test]
fn rejects_invalid_input() {
    let mut cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/parity.json")).expect("fixtures");
    let mut req = cases.remove(2).request;
    req.obligors[0].pd = 0.0;
    assert!(pykhtin::analyze(&req).unwrap_err().contains("pd"));
    req.obligors[0].pd = 0.01;
    req.confidence_level = 1.5;
    assert!(pykhtin::analyze(&req).unwrap_err().contains("confidence_level"));
}
