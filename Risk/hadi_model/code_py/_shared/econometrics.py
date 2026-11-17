"""Econometric helpers that pyfixest does not provide.

* ``conley_vcov``  Conley (1999) spatial HAC covariance, the analogue of
  ``fixest::conley(cutoff = 100)``.
* ``avg_slopes``   average marginal effects with delta-method SEs, the analogue
  of ``marginaleffects::avg_slopes(model, variables = ...)`` for models whose
  nonlinearity is limited to squares ``I(x ** 2)`` and products ``a:b:c``.
* ``FitResult``    a light, picklable summary of a fitted OLS model. It stands in
  for the ``fixest`` object that the R code writes to ``.rds``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse, stats

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    name: str
    formula: str
    vcov_type: str
    coefnames: list[str]
    coef: np.ndarray
    vcov: np.ndarray
    nobs: int
    r2: float
    extra: dict = field(default_factory=dict)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.vcov))

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.coef)

    def tidy(self) -> pd.DataFrame:
        t = self.coef / self.se
        p = 2 * stats.t.sf(np.abs(t), df=self.df_resid)
        crit = stats.t.ppf(0.975, df=self.df_resid)
        return pd.DataFrame(
            {
                "term": self.coefnames,
                "estimate": self.coef,
                "std_error": self.se,
                "statistic": t,
                "p_value": p,
                "conf_low": self.coef - crit * self.se,
                "conf_high": self.coef + crit * self.se,
                "vcov_type": self.vcov_type,
                "nobs": self.nobs,
            }
        )


# --------------------------------------------------------------------------- #
# Conley spatial HAC
# --------------------------------------------------------------------------- #
def conley_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cutoff_km: float = 100.0,
    chunk_size: int = 20_000,
    small_sample_adj: bool = True,
) -> np.ndarray:
    """Conley spatial HAC variance-covariance matrix for OLS.

    V = (X'X)^-1  [ sum_i sum_j K(d_ij) (x_i u_i)(x_j u_j)' ]  (X'X)^-1

    with a uniform kernel, K = 1 when the great-circle distance d_ij <= cutoff and
    0 otherwise, as in ``fixest::conley``. Neighbours come from a haversine BallTree
    radius query, processed in chunks so the full N x N weight matrix is never
    held in memory.

    ``small_sample_adj`` multiplies by (n - 1) / (n - k), the default fixest
    small-sample correction ``ssc(adj = TRUE)``.
    """
    from sklearn.neighbors import BallTree

    X = np.asarray(X, dtype=float)
    u = np.asarray(resid, dtype=float).ravel()
    n, k = X.shape
    scores = X * u[:, None]  # s_i = x_i * u_i

    coords = np.radians(np.column_stack([lat, lon]).astype(float))
    tree = BallTree(coords, metric="haversine")
    radius = cutoff_km / EARTH_RADIUS_KM

    meat = np.zeros((k, k))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        neigh = tree.query_radius(coords[start:stop], r=radius)
        lens = np.fromiter((len(a) for a in neigh), dtype=np.int64, count=len(neigh))
        indptr = np.concatenate([[0], np.cumsum(lens)])
        indices = np.concatenate(neigh) if len(neigh) else np.empty(0, np.int64)
        A = sparse.csr_matrix(
            (np.ones(indices.size), indices, indptr), shape=(stop - start, n)
        )
        neighbour_sum = A @ scores  # sum_{j in N(i)} s_j
        meat += scores[start:stop].T @ neighbour_sum

    meat = (meat + meat.T) / 2  # remove floating-point asymmetry
    bread = np.linalg.inv(X.T @ X)
    V = bread @ meat @ bread
    if small_sample_adj:
        V *= (n - 1) / (n - k)
    return V


# --------------------------------------------------------------------------- #
# Average marginal effects (marginaleffects::avg_slopes)
# --------------------------------------------------------------------------- #
_SQ_RE = re.compile(r"^I\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_.]*)\s*(\*\*|\^)\s*2\s*\)$")


def _factor_value(factor: str, data: pd.DataFrame) -> np.ndarray:
    m = _SQ_RE.match(factor)
    if m:
        return data[m.group("var")].to_numpy(float) ** 2
    return data[factor].to_numpy(float)


def _term_derivative(term: str, var: str, data: pd.DataFrame) -> np.ndarray | None:
    """d(term)/d(var) per row. Returns None when the term does not involve var.

    Handles the plain variable, ``I(var ** 2)``, and ``:`` products of these.
    """
    factors = term.split(":")
    involved = [
        i for i, f in enumerate(factors)
        if f == var or (_SQ_RE.match(f) and _SQ_RE.match(f).group("var") == var)
    ]
    if not involved:
        return None
    total = np.zeros(len(data))
    for i in involved:
        f = factors[i]
        d = 2 * data[var].to_numpy(float) if f != var else np.ones(len(data))
        for j, other in enumerate(factors):
            if j != i:
                d = d * _factor_value(other, data)
        total += d
    return total


def avg_slopes(fit: FitResult, data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Average marginal effect of each variable, with delta-method SEs.

    AME_v = mean_i  d yhat_i / d x_{iv} = gbar_v' beta,  where
    gbar_v = mean_i  d(design row_i)/d x_{iv}.

    The AME is linear in beta, so the delta-method variance is exactly
    gbar_v' V gbar_v. ``data`` must be the estimation sample (same rows as the fit).
    """
    rows = []
    for v in variables:
        g = np.zeros(len(fit.coef))
        for c, term in enumerate(fit.coefnames):
            d = _term_derivative(term, v, data)
            if d is not None:
                g[c] = d.mean()
        est = float(g @ fit.coef)
        se = float(np.sqrt(g @ fit.vcov @ g))
        z = est / se
        rows.append(
            {
                "term": v,
                "estimate": est,
                "std_error": se,
                "statistic": z,
                "p_value": 2 * stats.norm.sf(abs(z)),
                "conf_low": est - 1.959963984540054 * se,
                "conf_high": est + 1.959963984540054 * se,
                "vcov_type": fit.vcov_type,
            }
        )
    return pd.DataFrame(rows)
