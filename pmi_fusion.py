"""
Postmortem Interval (PMI) Estimation — Bayesian Fusion Core
============================================================
Three evidence streams fused into a single posterior distribution:
  Layer 1 — Henssge double-exponential body cooling model (1988)
  Layer 2 — Livor/rigor mortis stage likelihood distributions
  Layer 3 — Bayesian multiplication + posterior credible intervals

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PMI_GRID_MAX   = 72.0   # hours — upper bound of estimation window
PMI_GRID_STEP  = 0.25   # hours — resolution of the discrete PMI grid
T_DEATH        = 37.2   # °C — assumed core temperature at time of death

PMI_GRID = np.arange(0, PMI_GRID_MAX + PMI_GRID_STEP, PMI_GRID_STEP)


# ---------------------------------------------------------------------------
# Layer 1 — Henssge double-exponential cooling model
# ---------------------------------------------------------------------------

def henssge_k(mass_kg: float) -> float:
    """Empirical cooling rate constant for body mass (Henssge 1988)."""
    return 0.0250 * np.exp(-0.164 * (mass_kg / 70.0) ** 0.386)


def henssge_temp(t_hours: float, t_amb: float, mass_kg: float, cf: float) -> float:
    """
    Predicted rectal temperature at time t post-mortem.
    Uses the full double-exponential plateau correction.
    """
    k = henssge_k(mass_kg) * cf
    ratio = 1.25 * np.exp(-k * t_hours) - 0.25 * np.exp(-5 * k * t_hours)
    return t_amb + (T_DEATH - t_amb) * ratio


def henssge_invert(t_body: float, t_amb: float, mass_kg: float, cf: float) -> float:
    """
    Numerically invert the Henssge equation to find the PMI that
    produces the observed body temperature. Uses Newton-Raphson.
    Returns PMI in hours; clamps to [0, PMI_GRID_MAX].
    """
    if t_body <= t_amb:
        return PMI_GRID_MAX   # fully cooled — PMI beyond window
    if t_body >= T_DEATH:
        return 0.0             # still at death temp — very recent

    k = henssge_k(mass_kg) * cf
    target = (t_body - t_amb) / (T_DEATH - t_amb)

    # Newton-Raphson starting from a simple single-exponential guess
    t = -np.log(target) / k
    for _ in range(50):
        f  = 1.25 * np.exp(-k * t) - 0.25 * np.exp(-5 * k * t) - target
        df = k * (-1.25 * np.exp(-k * t) + 1.25 * np.exp(-5 * k * t))
        if abs(df) < 1e-12:
            break
        t -= f / df
        t = max(0.0, min(t, PMI_GRID_MAX))

    return float(t)


def henssge_likelihood(
    t_body: float,
    t_amb: float,
    mass_kg: float,
    cf: float,
    sigma_base: float = 2.0,
) -> np.ndarray:
    """
    Gaussian likelihood over the PMI grid centred on the Henssge inversion.
    """
    mu = henssge_invert(t_body, t_amb, mass_kg, cf)

    temp_diff = T_DEATH - t_amb
    sigma = sigma_base
    sigma += 0.06 * t_amb                      # warmer ambient → more uncertainty
    sigma += 1.5 * abs(cf - 1.0)               # corrective factor uncertainty
    sigma += max(0, 2.0 - temp_diff / 8.0)     # small differential → wider

    likelihood = norm.pdf(PMI_GRID, loc=mu, scale=max(sigma, 0.5))
    return _normalise(likelihood)


# ---------------------------------------------------------------------------
# Layer 2 — Postmortem change likelihoods
# ---------------------------------------------------------------------------

@dataclass
class StageWindow:
    """Asymmetric Gaussian PMI window for a postmortem change stage."""
    mu: float           # peak probability (hours)
    sigma_lo: float     # spread below mu
    sigma_hi: float     # spread above mu
    weight: float = 1.0 # relative confidence in this method


LIVOR_WINDOWS = {
    "none":           StageWindow(0.5,  0.5,  1.0),
    "faint":          StageWindow(3.0,  2.0,  3.0),
    "confluent":      StageWindow(9.0,  4.0,  5.0),
    "fixed":          StageWindow(26.0, 8.0,  10.0),
}

RIGOR_WINDOWS = {
    "none":           StageWindow(1.0,  1.0,  1.5),
    "partial":        StageWindow(5.0,  3.0,  4.0),
    "full":           StageWindow(13.0, 5.0,  5.0),
    "resolving":      StageWindow(36.0, 8.0,  12.0),
}


def _asymmetric_gaussian(mu: float, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    sigma = np.where(PMI_GRID < mu, sigma_lo, sigma_hi)
    return np.exp(-0.5 * ((PMI_GRID - mu) / sigma) ** 2)


def stage_likelihood(stage_key: str, windows: dict) -> np.ndarray:
    """
    Return a normalised PMI likelihood for a named postmortem change stage.
    """
    if stage_key not in windows:
        raise ValueError(f"Unknown stage '{stage_key}'. Valid: {list(windows)}")
    w = windows[stage_key]
    raw = _asymmetric_gaussian(w.mu, w.sigma_lo, w.sigma_hi)
    return _normalise(raw)


def image_likelihood(
    livor_stage: str,
    rigor_stage:  str,
    livor_weight: float = 1.0,
    rigor_weight: float = 1.0,
) -> np.ndarray:
    """
    Combined image-based likelihood from livor + rigor observations.
    """
    l_like = stage_likelihood(livor_stage, LIVOR_WINDOWS) ** livor_weight
    r_like = stage_likelihood(rigor_stage, RIGOR_WINDOWS) ** rigor_weight
    return _normalise(l_like * r_like)


def image_likelihood_from_softmax(
    livor_softmax: dict[str, float],
    rigor_softmax: dict[str, float],
) -> np.ndarray:
    """
    CNN-ready variant: accepts full softmax probability vectors over stages.
    """
    livor_mix = np.zeros_like(PMI_GRID)
    for stage, prob in livor_softmax.items():
        livor_mix += prob * stage_likelihood(stage, LIVOR_WINDOWS)

    rigor_mix = np.zeros_like(PMI_GRID)
    for stage, prob in rigor_softmax.items():
        rigor_mix += prob * stage_likelihood(stage, RIGOR_WINDOWS)

    return _normalise(livor_mix * rigor_mix)


# ---------------------------------------------------------------------------
# Layer 3 — Bayesian fusion
# ---------------------------------------------------------------------------

def bayesian_fuse(*likelihoods: np.ndarray) -> np.ndarray:
    """
    Multiply independent likelihood arrays element-wise.
    Returns the normalised posterior over PMI_GRID.
    """
    posterior = np.ones_like(PMI_GRID)
    for lik in likelihoods:
        posterior = posterior * lik
    return _normalise(posterior)


# ---------------------------------------------------------------------------
# Posterior analysis
# ---------------------------------------------------------------------------

@dataclass
class PMIResult:
    mode_hr: float
    mean_hr: float
    ci_68: tuple[float, float]
    ci_95: tuple[float, float]
    conflict: bool
    conflict_spread_hr: float
    stream_modes: dict[str, float]
    posterior: np.ndarray = field(repr=False)
    pmi_grid: np.ndarray  = field(repr=False)

    def summary(self) -> str:
        ci68 = self.ci_68
        ci95 = self.ci_95
        lines = [
            f"PMI mode estimate  : {self.mode_hr:.1f} hr",
            f"PMI mean estimate  : {self.mean_hr:.1f} hr",
            f"68% credible int.  : {ci68[0]:.1f} – {ci68[1]:.1f} hr",
            f"95% credible int.  : {ci95[0]:.1f} – {ci95[1]:.1f} hr",
            f"Stream modes       : { {k: f'{v:.1f}h' for k, v in self.stream_modes.items()} }",
            f"Evidence conflict  : {'YES — spread {:.1f} hr — FLAG FOR REVIEW'.format(self.conflict_spread_hr) if self.conflict else 'No ({:.1f} hr spread)'.format(self.conflict_spread_hr)}",
        ]
        return "\n".join(lines)


def credible_interval(posterior: np.ndarray, prob: float) -> tuple[float, float]:
    """
    Compute the highest-density credible interval at probability `prob`.
    """
    cdf = np.cumsum(posterior) * PMI_GRID_STEP
    lo_mass = (1 - prob) / 2
    hi_mass = 1 - lo_mass

    lo_idx = np.searchsorted(cdf, lo_mass)
    hi_idx = np.searchsorted(cdf, hi_mass)
    lo_idx = min(lo_idx, len(PMI_GRID) - 1)
    hi_idx = min(hi_idx, len(PMI_GRID) - 1)

    return float(PMI_GRID[lo_idx]), float(PMI_GRID[hi_idx])


def _mode(arr: np.ndarray) -> float:
    return float(PMI_GRID[np.argmax(arr)])


def _mean(arr: np.ndarray) -> float:
    return float(np.sum(PMI_GRID * arr) * PMI_GRID_STEP)


def _normalise(arr: np.ndarray) -> np.ndarray:
    total = np.sum(arr) * PMI_GRID_STEP
    if total < 1e-300:
        return np.ones_like(arr) / (len(arr) * PMI_GRID_STEP)
    return arr / total


CONFLICT_THRESHOLD_HR = 6.0   # flag if stream modes diverge by more than this


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_pmi(
    t_body_celsius:    float,
    t_ambient_celsius: float,
    body_mass_kg:      float,
    corrective_factor: float = 1.0,
    henssge_sigma:     float = 2.0,
    livor_stage: Optional[str] = None,
    rigor_stage:  Optional[str] = None,
    livor_softmax: Optional[dict[str, float]] = None,
    rigor_softmax: Optional[dict[str, float]] = None,
    use_henssge: bool = True,
    use_image:   bool = True,
) -> PMIResult:
    """
    Fuse all available evidence streams into a single PMI posterior.
    """
    likelihoods = []
    stream_modes = {}

    # --- Layer 1: Henssge ---
    if use_henssge:
        h_lik = henssge_likelihood(
            t_body_celsius, t_ambient_celsius, body_mass_kg,
            corrective_factor, henssge_sigma
        )
        likelihoods.append(h_lik)
        stream_modes["henssge"] = _mode(h_lik)

    # --- Layer 2: Image / postmortem changes ---
    if use_image:
        if livor_softmax is not None and rigor_softmax is not None:
            im_lik = image_likelihood_from_softmax(livor_softmax, rigor_softmax)
        elif livor_stage is not None and rigor_stage is not None:
            im_lik = image_likelihood(livor_stage, rigor_stage)
        else:
            im_lik = None

        if im_lik is not None:
            likelihoods.append(im_lik)
            stream_modes["image"] = _mode(im_lik)

    if not likelihoods:
        raise ValueError("At least one evidence stream must be enabled.")

    # --- Layer 3: Bayesian fusion ---
    posterior = bayesian_fuse(*likelihoods)

    # --- Conflict detection ---
    modes_arr = list(stream_modes.values())
    spread = max(modes_arr) - min(modes_arr) if len(modes_arr) > 1 else 0.0
    conflict = spread > CONFLICT_THRESHOLD_HR

    return PMIResult(
        mode_hr           = _mode(posterior),
        mean_hr           = _mean(posterior),
        ci_68             = credible_interval(posterior, 0.68),
        ci_95             = credible_interval(posterior, 0.95),
        conflict          = conflict,
        conflict_spread_hr= spread,
        stream_modes      = stream_modes,
        posterior         = posterior,
        pmi_grid          = PMI_GRID,
    )


HENSSGE_CF_TABLE = {
    "naked_air":              1.00,
    "1_2_layers_clothing":    0.75,
    "3_4_layers_clothing":    0.50,
    "heavy_clothing":         0.35,
    "water_still":            1.45,
    "water_flowing":          1.60,
    "air_moving":             1.20,
}


if __name__ == "__main__":

    print("=" * 60)
    print("Example 1 — Standard indoor scene (hard stage labels)")
    print("=" * 60)
    result = estimate_pmi(
        t_body_celsius    = 31.5,
        t_ambient_celsius = 21.0,
        body_mass_kg      = 75.0,
        corrective_factor = HENSSGE_CF_TABLE["1_2_layers_clothing"],
        livor_stage       = "confluent",
        rigor_stage       = "partial",
    )
    print(result.summary())

    print()
    print("=" * 60)
    print("Example 2 — CNN softmax inputs")
    print("=" * 60)
    result2 = estimate_pmi(
        t_body_celsius    = 28.0,
        t_ambient_celsius = 18.0,
        body_mass_kg      = 90.0,
        corrective_factor = 1.0,
        livor_softmax = {
            "none": 0.02, "faint": 0.08, "confluent": 0.75, "fixed": 0.15
        },
        rigor_softmax = {
            "none": 0.01, "partial": 0.20, "full": 0.72, "resolving": 0.07
        },
    )
    print(result2.summary())

    print()
    print("=" * 60)
    print("Example 3 — Evidence conflict scenario")
    print("=" * 60)
    result3 = estimate_pmi(
        t_body_celsius    = 35.5,
        t_ambient_celsius = 22.0,
        body_mass_kg      = 68.0,
        corrective_factor = 1.0,
        livor_stage       = "fixed",
        rigor_stage       = "full",
    )
    print(result3.summary())

    print()
    print("=" * 60)
    print("Example 4 — Henssge only")
    print("=" * 60)
    result4 = estimate_pmi(
        t_body_celsius    = 34.0,
        t_ambient_celsius = 15.0,
        body_mass_kg      = 82.0,
        corrective_factor = HENSSGE_CF_TABLE["water_still"],
        use_image         = False,
    )
    print(result4.summary())
