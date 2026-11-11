import json
from pathlib import Path

import numpy as np
import pytest

import cpm_approx
from cpm_approx import analyze, paper_portfolio, portfolio_risk

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures" / "parity.json").read_text())

# (rho, counts, VaR %, ES %) from the "Approx" columns of Table C in the paper
TABLE_C = [
    (0.5, "I", 2.33, 2.76), (0.5, "II", 3.06, 3.55), (0.5, "III", 2.32, 2.77),
    (0.3, "I", 1.90, 2.18), (0.3, "II", 2.80, 3.15), (0.1, "I", 1.55, 1.71), (0.1, "II", 2.82, 3.09),
]
COUNTS = {"I": [50, 100, 50, 100, 50, 100, 50, 100, 50, 100],
          "II": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
          "III": [10, 20, 50, 50, 100, 100, 200, 200, 500, 1000]}


@pytest.mark.parametrize("rho,counts,var,es", TABLE_C)
def test_reproduces_table_c(rho, counts, var, es):
    r = portfolio_risk(**paper_portfolio(rho, COUNTS[counts]), confidence=0.999)
    assert r.expected_loss == pytest.approx(0.00451, abs=2e-5)
    assert 100 * r.var == pytest.approx(var, abs=0.015)
    assert 100 * r.es == pytest.approx(es, abs=0.015)


def test_limit_and_ordering():
    r = portfolio_risk(**paper_portfolio(0.5), confidence=0.999)
    assert 100 * r.var_inf == pytest.approx(2.15, abs=0.015)
    assert r.expected_loss < r.var_limit < r.var_inf < r.var < r.es


@pytest.mark.parametrize("case", FIXTURES)
def test_matches_python_reference(case):
    out = analyze(case["request"])
    for key, expected in case["expected"].items():
        assert out[key] == pytest.approx(expected, rel=1e-9, abs=1e-14), key


def test_numpy_input_and_single_factor_default():
    n = 50
    rng = np.random.default_rng(0)
    pd = rng.uniform(0.005, 0.05, n)
    r = portfolio_risk(pd, np.full(n, 0.45), rng.uniform(1, 3, n), np.full(n, 0.2), 0.99)
    assert r.expected_loss < r.var < r.es < 1
    same = portfolio_risk(list(pd), [0.45] * n, list(rng.uniform(1, 3, n)), [0.2] * n, 0.99)
    assert same.var > 0


def test_weights_with_factor_correlation_equals_loadings():
    m = 20
    corr = [[1.0, 0.3], [0.3, 1.0]]
    weights = [[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(m)]
    kwargs = dict(pd=[0.02] * m, lgd=[0.45] * m, exposure=[1.0] * m, asset_correlation=[0.2] * m, confidence=0.99)
    a = portfolio_risk(**kwargs, factor_weights=weights, factor_correlation=corr)
    chol = np.linalg.cholesky(corr)
    loadings = [(np.array(w) @ chol) / np.linalg.norm(np.array(w) @ chol) for w in weights]
    b = portfolio_risk(**kwargs, loadings=loadings)
    assert a.var == pytest.approx(b.var, rel=1e-12)


def test_invalid_input_raises_value_error():
    ok = dict(pd=[0.01], lgd=[0.4], exposure=[1.0], asset_correlation=[0.2], confidence=0.99)
    with pytest.raises(ValueError, match="confidence"):
        portfolio_risk(**{**ok, "confidence": 1.5})
    with pytest.raises(ValueError, match="pd"):
        portfolio_risk(**{**ok, "pd": [0.0]})
    with pytest.raises(ValueError, match="same length"):
        portfolio_risk(**{**ok, "lgd": [0.4, 0.4]})
    with pytest.raises(ValueError, match="either"):
        portfolio_risk(**ok, factor_weights=[[1.0]], loadings=[[1.0]])
    with pytest.raises(ValueError, match="empty"):
        analyze({"confidence_level": 0.99, "obligors": []})


def test_version():
    assert cpm_approx.__version__ == "0.1.0"
