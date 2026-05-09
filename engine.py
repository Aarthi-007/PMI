"""
PMI Bayesian Fusion Engine
==========================
Postmortem interval estimation via three fused evidence streams.
  Layer 1 — Henssge double-exponential body cooling (1988)
  Layer 2 — Livor/rigor mortis stage likelihoods (hard labels or CNN softmax)
  Layer 3 — Bayesian posterior fusion + credible intervals

Requires: numpy, scipy
"""

import numpy as np
from scipy.stats import norm as sp_norm
from dataclasses import dataclass, field
from typing import Optional

# ── Grid ────────────────────────────────────────────────────────────────────
PMI_MAX  = 72.0   # hours — estimation window upper bound
PMI_STEP = 0.25   # hours — grid resolution
PMI_GRID = np.arange(0, PMI_MAX + PMI_STEP, PMI_STEP)

T_DEATH  = 37.2   # °C — assumed core temperature at time of death
CONFLICT_THRESHOLD = 6.0  # hours — flag if stream modes diverge beyond this


# ── Henssge corrective factors (Table 3.1, Henssge 1988) ────────────────────
CORRECTIVE_FACTORS = {
    "naked_air":           1.00,
    "1_2_layers":          0.75,
    "3_4_layers":          0.50,
    "heavy_clothing":      0.35,
    "water_still":         1.45,
    "water_flowing":       1.60,
    "moving_air":          1.20,
}


# ── Stage windows ─────────────────────────────────────────────────────────
# Asymmetric Gaussians: (peak_hr, sigma_lo, sigma_hi)
# Sources: DiMaio & DiMaio 2001, Knight 1996, Dolinak et al. 2005
LIVOR_WINDOWS = {
    "none":      (0.5,   0.5,  1.0),
    "faint":     (3.0,   2.0,  3.0),
    "confluent": (9.0,   4.0,  5.0),
    "fixed":     (26.0,  8.0, 10.0),
}
RIGOR_WINDOWS = {
    "none":      (1.0,   1.0,  1.5),
    "partial":   (5.0,   3.0,  4.0),
    "full":      (13.0,  5.0,  5.0),
    "resolving": (36.0,  8.0, 12.0),
}


# ── Utilities ────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    total = np.sum(arr) * PMI_STEP
    if total < 1e-300:
        return np.ones_like(arr) / (len(arr) * PMI_STEP)
    return arr / total

def _mode(arr: np.ndarray) -> float:
    return float(PMI_GRID[np.argmax(arr)])

def _mean(arr: np.ndarray) -> float:
    return float(np.sum(PMI_GRID * arr) * PMI_STEP)

def _credible_interval(posterior: np.ndarray, prob: float):
    cdf = np.cumsum(posterior) * PMI_STEP
    lo = (1 - prob) / 2
    hi = 1 - lo
    lo_idx = min(np.searchsorted(cdf, lo), len(PMI_GRID) - 1)
    hi_idx = min(np.searchsorted(cdf, hi), len(PMI_GRID) - 1)
    return float(PMI_GRID[lo_idx]), float(PMI_GRID[hi_idx])

def _asym_gauss(mu, sigma_lo, sigma_hi) -> np.ndarray:
    sigma = np.where(PMI_GRID < mu, sigma_lo, sigma_hi)
    return np.exp(-0.5 * ((PMI_GRID - mu) / sigma) ** 2)


# ── Layer 1: Henssge ─────────────────────────────────────────────────────────

def _henssge_k(mass_kg: float) -> float:
    """Empirical cooling rate constant (Henssge 1988 closed form)."""
    return 0.0250 * np.exp(-0.164 * (mass_kg / 70.0) ** 0.386)

def _henssge_invert(t_body, t_amb, mass_kg, cf) -> float:
    """Newton-Raphson inversion of double-exponential to find PMI."""
    if t_body <= t_amb:
        return PMI_MAX
    if t_body >= T_DEATH:
        return 0.0
    k = _henssge_k(mass_kg) * cf
    target = (t_body - t_amb) / (T_DEATH - t_amb)
    t = -np.log(max(target, 1e-9)) / k
    for _ in range(60):
        f  = 1.25 * np.exp(-k*t) - 0.25 * np.exp(-5*k*t) - target
        df = k * (-1.25 * np.exp(-k*t) + 1.25 * np.exp(-5*k*t))
        if abs(df) < 1e-12:
            break
        t -= f / df
        t = max(0.0, min(t, PMI_MAX))
    return float(t)

def henssge_likelihood(t_body, t_amb, mass_kg, cf=1.0, sigma_base=2.0) -> np.ndarray:
    """
    Gaussian likelihood over PMI_GRID centred on Henssge inversion.
    Uncertainty scales with ambient temp, corrective-factor deviation,
    and body-ambient differential.
    """
    mu = _henssge_invert(t_body, t_amb, mass_kg, cf)
    sigma = sigma_base
    sigma += 0.06 * t_amb
    sigma += 1.5  * abs(cf - 1.0)
    sigma += max(0, 2.0 - (T_DEATH - t_amb) / 8.0)
    raw = sp_norm.pdf(PMI_GRID, loc=mu, scale=max(sigma, 0.5))
    return _normalise(raw)


# ── Layer 2: Postmortem change likelihoods ────────────────────────────────────

def _stage_likelihood(stage: str, windows: dict) -> np.ndarray:
    if stage not in windows:
        raise ValueError(f"Unknown stage '{stage}'. Valid: {list(windows)}")
    mu, s_lo, s_hi = windows[stage]
    return _normalise(_asym_gauss(mu, s_lo, s_hi))

def image_likelihood_hard(livor_stage: str, rigor_stage: str) -> np.ndarray:
    """Hard stage-label version (manual assessment)."""
    l = _stage_likelihood(livor_stage, LIVOR_WINDOWS)
    r = _stage_likelihood(rigor_stage,  RIGOR_WINDOWS)
    return _normalise(l * r)

def image_likelihood_softmax(
    livor_softmax: dict[str, float],
    rigor_softmax: dict[str, float],
) -> np.ndarray:
    """
    CNN-ready: weighted mixture over all stages.
    Preserves classifier uncertainty instead of collapsing to argmax.
    """
    livor_mix = sum(p * _stage_likelihood(s, LIVOR_WINDOWS)
                    for s, p in livor_softmax.items())
    rigor_mix = sum(p * _stage_likelihood(s, RIGOR_WINDOWS)
                    for s, p in rigor_softmax.items())
    return _normalise(livor_mix * rigor_mix)


# ── Layer 3: Bayesian fusion ──────────────────────────────────────────────────

def bayesian_fuse(*likelihoods: np.ndarray) -> np.ndarray:
    """Multiply independent likelihoods (uniform prior). Returns posterior."""
    posterior = np.ones_like(PMI_GRID)
    for lik in likelihoods:
        posterior = posterior * lik
    return _normalise(posterior)


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class PMIResult:
    mode_hr:            float
    mean_hr:            float
    ci_68:              tuple[float, float]
    ci_95:              tuple[float, float]
    conflict:           bool
    conflict_spread_hr: float
    stream_modes:       dict[str, float]
    henssge_mu:         float
    posterior:          np.ndarray = field(repr=False)
    henssge_lik:        np.ndarray = field(repr=False)
    image_lik:          np.ndarray = field(repr=False)
    pmi_grid:           np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "mode_hr":            round(self.mode_hr, 2),
            "mean_hr":            round(self.mean_hr, 2),
            "ci_68":              [round(x, 2) for x in self.ci_68],
            "ci_95":              [round(x, 2) for x in self.ci_95],
            "conflict":           self.conflict,
            "conflict_spread_hr": round(self.conflict_spread_hr, 2),
            "stream_modes":       {k: round(v, 2) for k, v in self.stream_modes.items()},
            "henssge_mu":         round(self.henssge_mu, 2),
            "pmi_grid":           self.pmi_grid.tolist(),
            "posterior":          self.posterior.tolist(),
            "henssge_lik":        self.henssge_lik.tolist(),
            "image_lik":          self.image_lik.tolist(),
        }

    def summary(self) -> str:
        ci68, ci95 = self.ci_68, self.ci_95
        flag = (f"⚠ CONFLICT — spread {self.conflict_spread_hr:.1f} hr — "
                f"FLAG FOR INVESTIGATOR REVIEW") if self.conflict else \
               f"OK ({self.conflict_spread_hr:.1f} hr spread)"
        return "\n".join([
            f"  Mode PMI     : {self.mode_hr:.1f} hr",
            f"  Mean PMI     : {self.mean_hr:.1f} hr",
            f"  68% CI       : {ci68[0]:.1f} – {ci68[1]:.1f} hr",
            f"  95% CI       : {ci95[0]:.1f} – {ci95[1]:.1f} hr",
            f"  Stream modes : { {k: f'{v:.1f}h' for k, v in self.stream_modes.items()} }",
            f"  Evidence     : {flag}",
        ])


# ── Main API ─────────────────────────────────────────────────────────────────

def estimate_pmi(
    t_body_celsius:    float,
    t_ambient_celsius: float,
    body_mass_kg:      float,
    corrective_factor: float = 1.0,
    henssge_sigma:     float = 2.0,
    livor_stage:       Optional[str] = None,
    rigor_stage:       Optional[str] = None,
    livor_softmax:     Optional[dict[str, float]] = None,
    rigor_softmax:     Optional[dict[str, float]] = None,
    use_henssge:       bool = True,
    use_image:         bool = True,
) -> PMIResult:
    """
    Fuse all available evidence into a PMI posterior.

    Parameters
    ----------
    t_body_celsius      Rectal temperature at scene (°C)
    t_ambient_celsius   Scene ambient temperature (°C)
    body_mass_kg        Body mass (kg)
    corrective_factor   Henssge Cf — see CORRECTIVE_FACTORS dict
    henssge_sigma       Base cooling-model uncertainty (hours)
    livor_stage         One of: none / faint / confluent / fixed
    rigor_stage         One of: none / partial / full / resolving
    livor_softmax       CNN softmax dict {stage: probability}
    rigor_softmax       CNN softmax dict {stage: probability}
    use_henssge         Include cooling evidence stream
    use_image           Include visual postmortem change stream
    """
    likelihoods  = []
    stream_modes = {}
    h_lik  = np.ones_like(PMI_GRID) / (len(PMI_GRID) * PMI_STEP)
    im_lik = np.ones_like(PMI_GRID) / (len(PMI_GRID) * PMI_STEP)
    h_mu   = 0.0

    if use_henssge:
        h_lik = henssge_likelihood(
            t_body_celsius, t_ambient_celsius,
            body_mass_kg, corrective_factor, henssge_sigma
        )
        h_mu = _henssge_invert(
            t_body_celsius, t_ambient_celsius, body_mass_kg, corrective_factor
        )
        likelihoods.append(h_lik)
        stream_modes["henssge"] = _mode(h_lik)

    if use_image:
        if livor_softmax and rigor_softmax:
            im_lik = image_likelihood_softmax(livor_softmax, rigor_softmax)
        elif livor_stage and rigor_stage:
            im_lik = image_likelihood_hard(livor_stage, rigor_stage)
        else:
            im_lik = None

        if im_lik is not None:
            likelihoods.append(im_lik)
            stream_modes["image"] = _mode(im_lik)

    if not likelihoods:
        raise ValueError("At least one evidence stream must be enabled.")

    posterior = bayesian_fuse(*likelihoods)

    modes_arr = list(stream_modes.values())
    spread    = max(modes_arr) - min(modes_arr) if len(modes_arr) > 1 else 0.0

    return PMIResult(
        mode_hr            = _mode(posterior),
        mean_hr            = _mean(posterior),
        ci_68              = _credible_interval(posterior, 0.68),
        ci_95              = _credible_interval(posterior, 0.95),
        conflict           = spread > CONFLICT_THRESHOLD,
        conflict_spread_hr = spread,
        stream_modes       = stream_modes,
        henssge_mu         = h_mu,
        posterior          = posterior,
        henssge_lik        = h_lik,
        image_lik          = im_lik if im_lik is not None else
                             np.ones_like(PMI_GRID) / (len(PMI_GRID) * PMI_STEP),
        pmi_grid           = PMI_GRID,
    )
