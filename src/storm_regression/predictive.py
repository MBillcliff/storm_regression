"""
predictive.py — a uniform interface over per-window predictive distributions.

WHY THIS EXISTS
---------------
Every model path in the pipeline serialises its forecast differently:

    ensemble path : 'ensemble_predictions' (raw members) + 'log_mu'/'log_sigma'
                    + 'mu'/'sigma' (normal) + 'lambda'/'k' (weibull)
    mlp path      : 'log_mu'/'log_sigma' only
    set_mlp path  : 'mu_m'/'sigma_m'/'alpha' (per-member LogNormal mixture)

Without an abstraction, every *consumer* (dashboards, metrics, case studies) has to
re-implement "decode these particular keys -> compute a probability", so adding a model
means editing every consumer, and it is easy for two metrics to silently score two
different objects (e.g. CRPS on the raw ensemble while reliability uses the fitted
LogNormal).

This module inverts that: each model maps ONCE into a PredictiveForecast, and consumers
speak only the interface. CRPS and calibration are then guaranteed to describe the same
distribution, and a new model is a new subclass — nothing downstream changes.

SHAPES / CONVENTIONS
--------------------
Every forecast object represents a *batch* of B independent predictive distributions,
one per forecast window. Methods are vectorised over B:
    cdf(y)               -> (B,)     y is a scalar threshold or a (B,) array of obs
    quantile(q)          -> (B,)     q is a scalar probability in [0, 1]
    mean(), std()        -> (B,)
    median()             -> (B,)
    exceedance_prob(t)   -> (B,)     P(Y >= t)
    pit(y)               -> (B,)     CDF of each obs under its own forecast (calibration)
    interval(level)      -> (lo, hi) each (B,)   central prediction interval
    sample(n, rng)       -> (B, n)
    crps(y)              -> (B,)     lower is better
    subset(idx)          -> PredictiveForecast over the selected windows
"""

import numpy as np
from scipy.stats import norm, lognorm, weibull_min


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class PredictiveForecast:
    """Abstract per-window predictive distribution.

    Subclasses MUST implement `cdf`, `sample`, and `subset`. Everything else has a
    generic implementation here; subclasses override with closed forms where available.
    """

    # --- primitives subclasses must provide ---------------------------------
    def cdf(self, y):
        raise NotImplementedError

    def sample(self, n, rng=None):
        raise NotImplementedError

    def subset(self, idx):
        raise NotImplementedError

    # --- generic derived quantities -----------------------------------------
    def quantile(self, q, n=4000, rng=None):
        """Generic quantile via sampling. Parametric subclasses override with ppf."""
        s = self.sample(n, rng)
        return np.quantile(s, q, axis=1)

    def mean(self, n=4000, rng=None):
        return self.sample(n, rng).mean(axis=1)

    def std(self, n=4000, rng=None):
        return self.sample(n, rng).std(axis=1)

    def median(self):
        return self.quantile(0.5)

    def exceedance_prob(self, threshold):
        return 1.0 - self.cdf(threshold)

    def pit(self, y):
        """Probability integral transform: cdf of each observation under its forecast."""
        return self.cdf(np.asarray(y, dtype=float))

    def interval(self, level):
        """Central prediction interval covering `level` probability (e.g. 0.9)."""
        lo = self.quantile((1.0 - level) / 2.0)
        hi = self.quantile(1.0 - (1.0 - level) / 2.0)
        return lo, hi

    def crps(self, y, n_samples=2000, rng=None):
        """Generic sample-based CRPS (energy-score estimator). Lower is better.

        CRPS(F, y) = E|X - y| - 0.5 E|X - X'|,  X, X' iid ~ F.
        Parametric subclasses override with closed forms (no MC noise).
        """
        rng = np.random.default_rng(rng)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        X = self.sample(n_samples, rng)
        Xp = self.sample(n_samples, rng)
        term1 = np.mean(np.abs(X - y), axis=1)
        term2 = 0.5 * np.mean(np.abs(X - Xp), axis=1)
        return term1 - term2


# ---------------------------------------------------------------------------
# LogNormal  (the headline forecast for the mlp / ensemble paths)
# ---------------------------------------------------------------------------
class LogNormalForecast(PredictiveForecast):
    """LogNormal predictive distribution parameterised by log-space (mu, sigma).

    scipy convention: lognorm.cdf(y, s=sigma, scale=exp(mu)).
    """

    def __init__(self, log_mu, log_sigma):
        self.log_mu = np.asarray(log_mu, dtype=float)
        self.log_sigma = np.asarray(log_sigma, dtype=float)

    def cdf(self, y):
        return lognorm.cdf(y, s=self.log_sigma, scale=np.exp(self.log_mu))

    def quantile(self, q, **kw):
        return lognorm.ppf(q, s=self.log_sigma, scale=np.exp(self.log_mu))

    def mean(self, **kw):
        return np.exp(self.log_mu + 0.5 * self.log_sigma ** 2)

    def std(self, **kw):
        return self.mean() * np.sqrt(np.expm1(self.log_sigma ** 2))

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        eps = rng.standard_normal((self.log_mu.shape[0], n))
        return np.exp(self.log_mu[:, None] + self.log_sigma[:, None] * eps)

    def crps(self, y, **kw):
        """Closed-form CRPS for a LogNormal (Baran & Lerch, 2015). No MC noise."""
        y = np.maximum(np.asarray(y, dtype=float), 1e-9)
        s = self.log_sigma
        omega = (np.log(y) - self.log_mu) / s
        mean_ln = self.mean()
        return (y * (2.0 * norm.cdf(omega) - 1.0)
                - 2.0 * mean_ln * (norm.cdf(omega - s)
                                   + norm.cdf(s / np.sqrt(2.0)) - 1.0))

    def subset(self, idx):
        return LogNormalForecast(self.log_mu[idx], self.log_sigma[idx])


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------
class NormalForecast(PredictiveForecast):
    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)

    def cdf(self, y):
        return norm.cdf(y, loc=self.mu, scale=self.sigma)

    def quantile(self, q, **kw):
        return norm.ppf(q, loc=self.mu, scale=self.sigma)

    def mean(self, **kw):
        return self.mu.copy()

    def std(self, **kw):
        return self.sigma.copy()

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        eps = rng.standard_normal((self.mu.shape[0], n))
        return self.mu[:, None] + self.sigma[:, None] * eps

    def crps(self, y, **kw):
        """Closed-form CRPS for a Normal (Gneiting & Raftery, 2007)."""
        y = np.asarray(y, dtype=float)
        z = (y - self.mu) / self.sigma
        return self.sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                             + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

    def subset(self, idx):
        return NormalForecast(self.mu[idx], self.sigma[idx])


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------
class WeibullForecast(PredictiveForecast):
    """Weibull parameterised by (lambda=scale, k=shape).

    scipy convention: weibull_min.cdf(y, c=k, scale=lambda).
    """

    def __init__(self, lam, k):
        self.lam = np.asarray(lam, dtype=float)
        self.k = np.asarray(k, dtype=float)

    def cdf(self, y):
        return weibull_min.cdf(y, c=self.k, scale=self.lam)

    def quantile(self, q, **kw):
        return weibull_min.ppf(q, c=self.k, scale=self.lam)

    def mean(self, **kw):
        return weibull_min.mean(c=self.k, scale=self.lam)

    def std(self, **kw):
        return weibull_min.std(c=self.k, scale=self.lam)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        u = rng.random((self.lam.shape[0], n))
        return weibull_min.ppf(u, c=self.k[:, None], scale=self.lam[:, None])

    def subset(self, idx):
        return WeibullForecast(self.lam[idx], self.k[idx])


# ---------------------------------------------------------------------------
# Empirical ensemble  (raw member point-predictions)
# ---------------------------------------------------------------------------
class EnsembleForecast(PredictiveForecast):
    """Empirical predictive distribution from raw ensemble point predictions, (B, M)."""

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=float)  # (B, M)

    def cdf(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 0:
            return np.mean(self.samples <= y, axis=1)
        return np.mean(self.samples <= y[:, None], axis=1)

    def quantile(self, q, **kw):
        return np.quantile(self.samples, q, axis=1)

    def mean(self, **kw):
        return self.samples.mean(axis=1)

    def std(self, **kw):
        return self.samples.std(axis=1)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        B, M = self.samples.shape
        idx = rng.integers(0, M, size=(B, n))
        return np.take_along_axis(self.samples, idx, axis=1)

    def crps(self, y, **kw):
        """Exact empirical CRPS over the actual members (no resampling)."""
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        X = self.samples
        term1 = np.mean(np.abs(X - y), axis=1)
        # E|X - X'| over all member pairs; M ~ 100-200 so O(M^2) is fine
        term2 = 0.5 * np.mean(np.abs(X[:, :, None] - X[:, None, :]), axis=(1, 2))
        return term1 - term2

    def subset(self, idx):
        return EnsembleForecast(self.samples[idx])


# ---------------------------------------------------------------------------
# Mixture  (per-member LogNormal BMA — the set_mlp path, Phase 3)
# ---------------------------------------------------------------------------
class MixtureForecast(PredictiveForecast):
    """Weighted mixture of component forecasts: sum_k w_k F_k.

    components : list of K PredictiveForecast, each over the same batch B
    weights    : (B, K), rows sum to 1
    """

    def __init__(self, components, weights):
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=float)  # (B, K)

    def cdf(self, y):
        return sum(self.weights[:, k] * c.cdf(y)
                   for k, c in enumerate(self.components))

    def mean(self, **kw):
        return sum(self.weights[:, k] * c.mean()
                   for k, c in enumerate(self.components))

    def variance_decomposition(self):
        """Law of total variance: returns (within, between) variance, each (B,).

        within  = sum_k w_k Var_k            (aleatoric: spread within scenarios)
        between = sum_k w_k mu_k^2 - mu_bar^2 (scenario / ensemble uncertainty)
        """
        mu_bar = self.mean()
        within = sum(self.weights[:, k] * c.std() ** 2
                     for k, c in enumerate(self.components))
        between = sum(self.weights[:, k] * c.mean() ** 2
                      for k, c in enumerate(self.components)) - mu_bar ** 2
        return within, between

    def std(self, **kw):
        within, between = self.variance_decomposition()
        return np.sqrt(np.maximum(within + between, 0.0))

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        B = self.weights.shape[0]
        # draw component indices per (window, draw) from the per-window weights
        comp_samples = np.stack([c.sample(n, rng) for c in self.components], axis=1)  # (B,K,n)
        cum = np.cumsum(self.weights, axis=1)                                          # (B,K)
        u = rng.random((B, n))
        chosen = (u[:, None, :] > cum[:, :, None]).sum(axis=1)                         # (B,n) in [0,K-1]
        chosen = np.clip(chosen, 0, len(self.components) - 1)
        return np.take_along_axis(comp_samples, chosen[:, None, :], axis=1)[:, 0, :]

    def subset(self, idx):
        return MixtureForecast([c.subset(idx) for c in self.components],
                               self.weights[idx])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def forecast_from_results(results, family="auto"):
    """Build a PredictiveForecast from a results dict, regardless of how it was stored.

    Parameters
    ----------
    results : dict
        A results dict from any evaluation path.
    family : {'auto', 'lognormal', 'normal', 'weibull', 'ensemble', 'mixture'}
        Which predictive object to construct. 'auto' picks the headline forecast:
        mixture if present, else LogNormal, else the empirical ensemble.

    Notes
    -----
    The ensemble path stores LogNormal/Normal/Weibull params AND raw members
    simultaneously, so 'auto' deliberately prefers the parametric LogNormal (the
    forecast actually proposed) over the raw ensemble. Pass `family=` explicitly to
    score a specific representation (e.g. for a like-for-like comparison).
    """
    has = lambda *ks: all(k in results for k in ks)

    if family == "mixture" or (family == "auto" and has("mu_m", "sigma_m", "alpha")):
        mu_m, sig_m = np.asarray(results["mu_m"]), np.asarray(results["sigma_m"])
        comps = [LogNormalForecast(mu_m[:, m], sig_m[:, m]) for m in range(mu_m.shape[1])]
        return MixtureForecast(comps, results["alpha"])

    if family in ("auto", "lognormal") and has("log_mu", "log_sigma"):
        return LogNormalForecast(results["log_mu"], results["log_sigma"])

    if family == "normal" and has("mu", "sigma"):
        return NormalForecast(results["mu"], results["sigma"])

    if family == "weibull" and has("lambda", "k"):
        return WeibullForecast(results["lambda"], results["k"])

    if (family in ("auto", "ensemble")) and has("ensemble_predictions"):
        return EnsembleForecast(results["ensemble_predictions"])

    raise KeyError(
        f"Cannot build a '{family}' forecast from results with keys "
        f"{sorted(results.keys())}. Expected one of: "
        f"(mu_m,sigma_m,alpha) | (log_mu,log_sigma) | (mu,sigma) | (lambda,k) | "
        f"ensemble_predictions."
    )