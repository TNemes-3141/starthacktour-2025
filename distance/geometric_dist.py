"""
Monocular geometric distance estimator with modular uncertainties.

- No ML. Closed-form lognormal propagation and class-mixture aggregation.
- Efficient (O(C) per detection). Suitable for mobile (e.g., 10–15 objs < 30 ms).

Inputs per detection:
  - class probabilities (dict[str,float]) over candidate classes
  - bounding box (x1,y1,x2,y2) OR width/height in px
  - optional precomputed size measure in px (else computed from box)

Global inputs:
  - camera params (f_px OR HFOV + image_width)
  - priors for class physical sizes S_c (lognormal)
  - box tightness beta prior (lognormal), enable/disable
  - size measurement noise prior for s_i (lognormal), enable/disable
  - thresholds (s_min px, distance cap)

Outputs per detection (augment original dict):
  - distance_m_mode, distance_m_median, distance_m_mean
  - distance_m_ci90: [lo, hi]
  - per_class stats and the params used

All formulas follow the rigorous development shared previously.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import math

# ------------------------------
# Utilities: normal quantile (Acklam's approximation)
# ------------------------------

def _normal_ppf(p: float) -> float:
    """Approximate inverse CDF (quantile) of standard normal.
    Source: Peter J. Acklam's approximation.
    Valid for p in (0,1). Handles symmetry.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    # Coefficients in rational approximations
    a = [ -3.969683028665376e+01,
           2.209460984245205e+02,
          -2.759285104469687e+02,
           1.383577518672690e+02,
          -3.066479806614716e+01,
           2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,
           1.615858368580409e+02,
          -1.556989798598866e+02,
           6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03,
          -3.223964580411365e-01,
          -2.400758277161838e+00,
          -2.549732539343734e+00,
           4.374664141464968e+00,
           2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00 ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ------------------------------
# Data structures
# ------------------------------

SizeMeasure = Literal["max", "geom_mean", "mean"]

@dataclass
class CameraParams:
    # Either provide f_px directly, or HFOV (radians) and image width in px.
    f_px: Optional[float] = None
    hfov_rad: Optional[float] = None
    image_width_px: Optional[float] = None

    def effective_f_px(self) -> float:
        if self.f_px is not None:
            return float(self.f_px)
        if self.hfov_rad is None or self.image_width_px is None:
            raise ValueError("Provide either f_px or (hfov_rad and image_width_px)")
        if not (0 < self.hfov_rad < math.pi):
            raise ValueError("hfov_rad must be in (0, pi)")
        return self.image_width_px / (2.0 * math.tan(self.hfov_rad / 2.0))

@dataclass
class LogNormal:
    # lognormal with parameters on log-scale: ln X ~ N(mu_ln, sigma_ln^2)
    mu_ln: float
    sigma_ln: float

    @staticmethod
    def from_median_sigma_ln(median: float, sigma_ln: float) -> "LogNormal":
        if median <= 0:
            raise ValueError("median must be > 0")
        return LogNormal(mu_ln=math.log(median), sigma_ln=float(sigma_ln))

    def quantile(self, p: float) -> float:
        return math.exp(self.mu_ln + self.sigma_ln * _normal_ppf(p))

@dataclass
class ClassSizePriors:
    # map class name -> LogNormal prior of true physical span S_c (meters)
    priors: Dict[str, LogNormal]

    def get(self, cls: str) -> LogNormal:
        if cls not in self.priors:
            raise KeyError(f"Missing size prior for class '{cls}'")
        return self.priors[cls]

@dataclass
class TightnessPrior:
    enabled: bool = True
    # ln beta ~ N(mu_ln, sigma_ln^2), with median beta = exp(mu_ln) <= 1
    mu_ln: float = math.log(0.95)
    sigma_ln: float = 0.05

    def params(self) -> Tuple[float, float]:
        return (self.mu_ln if self.enabled else 0.0,
                self.sigma_ln if self.enabled else 0.0)

@dataclass
class SizeNoiseModel:
    enabled: bool = True
    # Either fixed sigma_ln, or function of (w,h) returning sigma_ln.
    fixed_sigma_ln: Optional[float] = None
    c0: float = 2.0  # used if fixed_sigma_ln is None: sigma_ln = min(0.15, c0/sqrt(w*h))
    cap: float = 0.15

    def sigma_ln_for_box(self, w: float, h: float) -> float:
        if not self.enabled:
            return 0.0
        if self.fixed_sigma_ln is not None:
            return float(self.fixed_sigma_ln)
        area = max(w * h, 1.0)
        return min(self.cap, self.c0 / math.sqrt(area))

@dataclass
class Heuristics:
    s_min_px: float = 6.0   # min effective px size gate
    distance_cap_m: float = 1000.0  # drop obviously distant (e.g., airliners)

@dataclass
class EstimatorConfig:
    size_measure: SizeMeasure = "geom_mean"
    class_size_priors: ClassSizePriors = field(default_factory=lambda: ClassSizePriors({}))
    tightness_prior: TightnessPrior = field(default_factory=TightnessPrior)
    size_noise: SizeNoiseModel = field(default_factory=SizeNoiseModel)
    heuristics: Heuristics = field(default_factory=Heuristics)
    ci_lower: float = 0.05
    ci_upper: float = 0.95

# ------------------------------
# Core math per class component
# ------------------------------

def _size_from_box(w: float, h: float, measure: SizeMeasure) -> float:
    w = max(float(w), 0.0)
    h = max(float(h), 0.0)
    if measure == "max":
        return max(w, h)
    if measure == "mean":
        return 0.5 * (w + h)
    # default geom_mean
    return math.sqrt(max(w * h, 0.0))

def _per_class_log_params(f_px: float,
                          s_px: float,
                          size_prior: LogNormal,
                          beta_mu_ln: float,
                          beta_sigma_ln: float,
                          s_sigma_ln: float) -> Tuple[float, float]:
    # ln D = ln f + ln S - ln beta - ln s
    if s_px <= 0:
        raise ValueError("s_px must be > 0")
    mu = math.log(f_px) + size_prior.mu_ln - beta_mu_ln - math.log(s_px)
    var = (size_prior.sigma_ln ** 2) + (beta_sigma_ln ** 2) + (s_sigma_ln ** 2)
    return mu, math.sqrt(var)

# ------------------------------
# Public API
# ------------------------------

def estimate_detection_distance(
    detection: Dict,
    camera: CameraParams,
    cfg: EstimatorConfig,
) -> Dict:
    """Augment a single detection dict with distance stats.

    Expected detection fields:
      - "bbox": [x1,y1,x2,y2] OR "w", "h"
      - "class_probs": {class_name: probability, ...} (need not be normalized)
      - optional: "size_px": precomputed size measure

    Returns a new dict with added keys and per-class breakdown.
    """
    out = dict(detection)  # shallow copy to augment

    # 1) Camera focal length in pixels
    f_px = camera.effective_f_px()

    # 2) Pixel size measure s_i
    if "size_px" in detection and detection["size_px"] is not None:
        s_px = float(detection["size_px"])
        # try to infer w,h for size noise if available
        w = float(detection.get("w", detection.get("width", 0.0)))
        h = float(detection.get("h", detection.get("height", 0.0)))
    else:
        if "w" in detection and "h" in detection:
            w = float(detection["w"]); h = float(detection["h"])
        elif "bbox" in detection:
            x1,y1,x2,y2 = map(float, detection["bbox"])
            w = max(x2 - x1, 0.0); h = max(y2 - y1, 0.0)
        else:
            raise KeyError("Detection must include (w,h) or bbox or size_px")
        s_px = _size_from_box(w, h, cfg.size_measure)

    # Heuristic gate for tiny detections
    if s_px < cfg.heuristics.s_min_px:
        out.update({
            "distance_m_mode": None,
            "distance_m_median": None,
            "distance_m_mean": None,
            "distance_m_ci90": None,
            "note": f"skipped: size_px<{cfg.heuristics.s_min_px}"
        })
        return out

    # 3) Class probabilities (normalize)
    cls_probs: Dict[str, float] = dict(detection.get("class_probs", {}))
    if not cls_probs:
        raise KeyError("Detection missing class_probs")
    # retain only classes with priors
    cls_probs = {c: p for (c, p) in cls_probs.items() if c in cfg.class_size_priors.priors}
    if not cls_probs:
        raise KeyError("No overlapping classes between detection and size priors")
    total_p = sum(max(p, 0.0) for p in cls_probs.values())
    if total_p <= 0:
        raise ValueError("class_probs must have positive mass")
    for c in list(cls_probs.keys()):
        cls_probs[c] = cls_probs[c] / total_p

    # 4) Uncertainty params
    beta_mu_ln, beta_sigma_ln = cfg.tightness_prior.params()
    s_sigma_ln = cfg.size_noise.sigma_ln_for_box(w, h) if cfg.size_noise.enabled else 0.0

    # 5) Per-class lognormal distance params and stats
    per_class_stats = []
    m1 = 0.0  # mixture mean accumulator
    m2 = 0.0  # mixture second moment accumulator
    best_cls = None
    best_p = -1.0
    best_mode = None

    for c, pc in cls_probs.items():
        size_prior = cfg.class_size_priors.get(c)
        mu_ln, sigma_ln = _per_class_log_params(
            f_px=f_px,
            s_px=s_px,
            size_prior=size_prior,
            beta_mu_ln=beta_mu_ln,
            beta_sigma_ln=beta_sigma_ln,
            s_sigma_ln=s_sigma_ln,
        )
        V = sigma_ln * sigma_ln
        # stats
        d_mode = math.exp(mu_ln - V)
        d_med = math.exp(mu_ln)
        d_mean = math.exp(mu_ln + 0.5 * V)
        q_lo = math.exp(mu_ln + _normal_ppf(cfg.ci_lower) * sigma_ln)
        q_hi = math.exp(mu_ln + _normal_ppf(cfg.ci_upper) * sigma_ln)

        # accumulate mixture moments
        m1 += pc * d_mean
        m2 += pc * math.exp(2.0 * mu_ln + 2.0 * V)

        if pc > best_p:
            best_p = pc
            best_cls = c
            best_mode = d_mode

        per_class_stats.append({
            "class": c,
            "p": pc,
            "mode": d_mode,
            "median": d_med,
            "mean": d_mean,
            "ci": [q_lo, q_hi],
            "ln_params": {"mu": mu_ln, "sigma": sigma_ln}
        })

    # 6) Mixture moment-match to a single lognormal
    # Guard against numerical issues
    m1 = max(m1, 1e-12)
    m2 = max(m2, m1*m1*(1+1e-12))
    V_star = math.log(m2 / (m1 * m1))
    mu_star = math.log(m1) - 0.5 * V_star

    # 7) Final stats
    z_lo = _normal_ppf(cfg.ci_lower)
    z_hi = _normal_ppf(cfg.ci_upper)
    d_median = math.exp(mu_star)
    d_mean = math.exp(mu_star + 0.5 * V_star)
    d_ci_lo = math.exp(mu_star + z_lo * math.sqrt(V_star))
    d_ci_hi = math.exp(mu_star + z_hi * math.sqrt(V_star))

    # 8) Optional distance cap filtering (do not drop, but annotate)
    note = None
    if d_median > cfg.heuristics.distance_cap_m:
        note = f"distance_median>{cfg.heuristics.distance_cap_m}m (likely far background)."

    out.update({
        "distance_m_mode": float(best_mode),  # MAP approx via top component mode
        "distance_m_median": float(d_median),
        "distance_m_mean": float(d_mean),
        "distance_m_ci90": [float(d_ci_lo), float(d_ci_hi)],
        "per_class": per_class_stats,
        "params_used": {
            "f_px": f_px,
            "size_measure": cfg.size_measure,
            "beta": {"enabled": cfg.tightness_prior.enabled,
                      "mu_ln": beta_mu_ln,
                      "sigma_ln": beta_sigma_ln},
            "s_noise": {"enabled": cfg.size_noise.enabled,
                         "sigma_ln": s_sigma_ln},
            "ci": [cfg.ci_lower, cfg.ci_upper]
        }
    })
    if note:
        out["note"] = note
    return out


def estimate_distances(
    detections: List[Dict],
    camera: CameraParams,
    cfg: EstimatorConfig,
) -> List[Dict]:
    """Vectorized wrapper over detections list.
    Returns a list of augmented detection dicts.
    """
    results: List[Dict] = []
    for det in detections:
        try:
            results.append(estimate_detection_distance(det, camera, cfg))
        except Exception as e:
            # Fail-closed per detection; annotate error
            d = dict(det)
            d["error"] = str(e)
            results.append(d)
    return results

# ------------------------------
# Example defaults & quick test harness (can be removed in production)
# ------------------------------

def default_priors() -> ClassSizePriors:
    return ClassSizePriors(priors={
        "bird":      LogNormal.from_median_sigma_ln(0.5, 0.40),
        "drone":     LogNormal.from_median_sigma_ln(0.30, 0.25),
        "paraglider":LogNormal.from_median_sigma_ln(10.0, 0.20),
        "balloon":   LogNormal.from_median_sigma_ln(17.0, 0.20),
        "helicopter":LogNormal.from_median_sigma_ln(10.0, 0.20),
        "light_ac":  LogNormal.from_median_sigma_ln(11.0, 0.20),
        "kite":      LogNormal.from_median_sigma_ln(1.5, 0.25),
        "rc_plane":  LogNormal.from_median_sigma_ln(1.0, 0.25),
        "parachute": LogNormal.from_median_sigma_ln(6.0, 0.20),
    })

if __name__ == "__main__":
    # Simple demo
    cam = CameraParams(hfov_rad=math.radians(65.0), image_width_px=4000)
    cfg = EstimatorConfig(
        size_measure="geom_mean",
        class_size_priors=default_priors(),
        tightness_prior=TightnessPrior(enabled=True, mu_ln=math.log(0.95), sigma_ln=0.05),
        size_noise=SizeNoiseModel(enabled=True, fixed_sigma_ln=None, c0=2.0, cap=0.15),
        heuristics=Heuristics(s_min_px=6.0, distance_cap_m=1000.0),
        ci_lower=0.05, ci_upper=0.95,
    )

    detections = [
        {"bbox": [100,100,124,116], "w":24, "h":16,
         "class_probs": {"bird":0.7, "drone":0.3}},
        {"bbox": [50,50,450,250], "w":400, "h":200,
         "class_probs": {"paraglider":0.9, "kite":0.1}},
    ]

    res = estimate_distances(detections, cam, cfg)
    for r in res:
        print({k:r[k] for k in ("distance_m_mode","distance_m_median","distance_m_ci90","note") if k in r})
