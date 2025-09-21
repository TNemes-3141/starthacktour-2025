# Monocular Range Estimator – Engineering Documentation

**Version:** 1.0
**Scope:** Non-ML, geometric distance estimation from a single RGB image (phone camera), with modular uncertainty and class-mixture handling.
**File:** `monocular_range_estimator.py`

---

## 1) What this library does

Given **detections** (class probabilities + bounding boxes) from a sky-facing camera, the library estimates the **slant distance** from the camera to each object. It:

* Uses the **pinhole camera** model to convert **pixel size** to **distance** using a **class-conditional physical size prior**.
* Models uncertainty in three **independent** modules (enable/disable per deployment needs):

  1. **Class size prior** (variation within a class, e.g., different bird sizes)
  2. **Bounding-box tightness** (box vs. true object silhouette)
  3. **Pixel size measurement noise** (detector jitter)
* Produces per-class **lognormal** distance distributions and combines them via a **class-probability mixture**.
* Returns: a **most-likely distance** (MAP approx) + a **credible range** (default 90% CI), plus per-class breakdown.
* Runs in **O(C)** operations per detection (C = candidate classes), suitable for mobile (10–15 objs in <30 ms).

**Non-goals:** No neural networks, no multi-view, no night-time support; intentionally simple, explainable physics + uncertainty propagation.

---

## 2) Core model in one picture

**Geometry:**
$D = \dfrac{f_{px} \cdot S}{\tilde{s}}$
Where:

* $D$: slant distance camera→object (m)
* $f_{px}$: focal length in **pixels** (derived from HFOV & image width or provided directly)
* $S$: true physical span of object (m), **class dependent**
* $\tilde{s} = \beta\, s$: effective pixel span; $s$ is measured from the bbox; $\beta\in(0,1] $ captures box looseness

**Uncertainties (log-space):**
$ \ln S \sim \mathcal{N}(\mu_{S,c}, \sigma_{S,c}^2)$, $\ln \beta \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$, $\ln s \sim \mathcal{N}(\ln \bar{s}, \sigma_{s}^2)$.
Then for each class $c$:

$\ln D \sim \mathcal{N}\big(\underbrace{\ln f_{px} + \mu_{S,c} - \mu_\beta - \ln \bar{s}}_{\mu_D}, \underbrace{\sigma_{S,c}^2 + \sigma_\beta^2 + \sigma_s^2}_{\sigma_D^2}\big).$

Per-class $D$ is lognormal; we compute **mode, median, mean, quantiles** in closed form. A weighted **mixture** over classes (weights = class probabilities) gives overall estimates; we moment-match to a single lognormal to get a fast CI.

---

## 3) Public API overview

```python
from monocular_range_estimator import (
    CameraParams, EstimatorConfig, ClassSizePriors, LogNormal,
    TightnessPrior, SizeNoiseModel, Heuristics,
    estimate_distances
)
```

### 3.1 Camera parameters

* **`CameraParams(f_px=None, hfov_rad=None, image_width_px=None)`**

  * Provide **either** `f_px` **or** `(hfov_rad, image_width_px)`.
  * `effective_f_px()` computes the pixel focal length.

### 3.2 Configuration

* **`EstimatorConfig`** fields:

  * `size_measure`: how pixel span is computed from bbox (`"geom_mean"` | `"max"` | `"mean"`).
  * `class_size_priors`: `ClassSizePriors({class_name: LogNormal})` in **meters**.
  * `tightness_prior`: `TightnessPrior(enabled, mu_ln=ln median β, sigma_ln)`; β≤1.
  * `size_noise`: `SizeNoiseModel(enabled, fixed_sigma_ln | (c0, cap))` → σ for `ln s`.
  * `heuristics`: `Heuristics(s_min_px, distance_cap_m)` gates tiny boxes & flags far background.
  * `ci_lower`, `ci_upper`: e.g., 0.05 and 0.95 for a 90% CI.

### 3.3 Estimation

* **`estimate_distances(detections, camera, cfg)`** → list of augmented detections.

  * Each detection must include **either** `(w,h)` **or** `bbox=[x1,y1,x2,y2]`, and `class_probs`.
  * Output adds: `distance_m_mode`, `distance_m_median`, `distance_m_mean`, `distance_m_ci90=[lo,hi]`, `per_class` breakdown, and `params_used`.

---

## 4) Input and output schema

### 4.1 Detection input

```json
{
  "bbox": [x1, y1, x2, y2],
  "w": 24,                // optional if bbox present
  "h": 16,                // optional if bbox present
  "class_probs": {"bird": 0.7, "drone": 0.3},
  "size_px": 20.0         // optional; overrides size measure from bbox
}
```

### 4.2 Augmented output (core fields)

```json
{
  "distance_m_mode": 123.4,            // most-likely (MAP approx)
  "distance_m_median": 128.7,
  "distance_m_mean": 131.2,
  "distance_m_ci90": [112.0, 149.5],   // [lo, hi]
  "per_class": [
    {"class":"bird","p":0.7,"mode":...,"median":...,"mean":...,"ci":[...],
     "ln_params":{"mu":...,"sigma":...}},
    {"class":"drone","p":0.3,...}
  ],
  "params_used": {
    "f_px": 2750.0,
    "size_measure": "geom_mean",
    "beta": {"enabled": true, "mu_ln": -0.0513, "sigma_ln": 0.05},
    "s_noise": {"enabled": true, "sigma_ln": 0.06},
    "ci": [0.05, 0.95]
  },
  "note": "distance_median>1000m (likely far background)." // optional
}
```

---

## 5) Configuration details & defaults

### 5.1 Class size priors (meters) – example

Use medians you trust; uncertainty width captures intra-class variability.

```python
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
```

**Tuning guidance:**

* Birds vary widely ⇒ larger `sigma_ln` (0.35–0.50).
* Manufactured/regulated classes (paraglider, balloon, light aircraft) ⇒ smaller `sigma_ln` (0.10–0.25).

### 5.2 Bounding-box tightness prior β

* `TightnessPrior(enabled=True, mu_ln=ln(0.95), sigma_ln=0.05)` by default.
* Interprets **effective** pixel span as `β * s`, where `s` comes from the bbox.
* If your detector is tight, set `enabled=False` or increase median β to \~0.98–1.0.

### 5.3 Pixel-size noise for ln(s)

* If `fixed_sigma_ln` is `None`, we use `sigma_ln = min(cap, c0 / sqrt(w*h))`.
* Defaults: `c0=2.0`, `cap=0.15` (≈ 15% relative noise cap). Increase cap for noisier detectors.

### 5.4 Heuristics

* `s_min_px` (default 6 px): small boxes are unreliable and often clouds/airliners → skipped.
* `distance_cap_m` (default 1000 m): flags when median distance is implausibly far; useful for logging/QA.

### 5.5 Credible interval

* `ci_lower=0.05`, `ci_upper=0.95` → 90% CI. Change to 0.025/0.975 for 95% CI.

---

## 6) Algorithmic steps (per detection)

1. **Compute pixel span** `s_px` from `(w,h)` via `size_measure` (or use provided `size_px`).
2. **Gate tiny objects**: if `s_px < s_min_px`, skip (unreliable).
3. **Normalize class probs**; drop classes without a size prior.
4. **Uncertainty params**: fetch `β` prior and `σ_ln` for `ln s`.
5. **Per-class distribution**: compute lognormal parameters `(μ_D, σ_D)` and stats (mode/median/mean/CI).
6. **Mixture aggregation**: accumulate mixture **mean** and **2nd moment**; moment-match to a single lognormal `(μ*, V*)` → overall median/mean/CI.
7. **Most-likely distance**: use the **mode** of the **top-probability** class as a MAP approximation (fast and robust).
8. **Annotate** if median exceeds `distance_cap_m`.

All operations are additions/multiplications and a few `log/exp`; quantiles use a fast normal-ppf approximation.

---

## 7) Performance characteristics

* Per class: \~a dozen float ops + 2–3 `exp/log`. Per detection: O(C).
* 15 detections × 8 classes = \~120 components → typically **well under 30 ms** on mid-range Android when running in optimized Python-equivalent (e.g., C++/Kotlin or PyPy on-device).
* No allocations on the hot path beyond small dicts; can be trivially ported to C++/Rust.

---

## 8) Examples

### 8.1 Minimal usage

```python
cam = CameraParams(hfov_rad=math.radians(65.0), image_width_px=4000)
cfg = EstimatorConfig(
    size_measure="geom_mean",
    class_size_priors=default_priors(),
)

detections = [
    {"bbox": [100,100,124,116], "w":24, "h":16,
     "class_probs": {"bird":0.7, "drone":0.3}},
    {"bbox": [50,50,450,250], "w":400, "h":200,
     "class_probs": {"paraglider":0.9, "kite":0.1}},
]

results = estimate_distances(detections, cam, cfg)
```

### 8.2 Tighter boxes, disable β and use fixed size noise

```python
cfg.tightness_prior.enabled = False
cfg.size_noise = SizeNoiseModel(enabled=True, fixed_sigma_ln=0.06)
```

### 8.3 Strict small-object gate

```python
cfg.heuristics.s_min_px = 8.0
```

---

## 9) Validation & calibration checklist

* **Sanity plots**: For each class, plot `D_median` vs. `1/s_px` across logs; expect linear trend (geometry).
* **Check β**: If estimates are systematically **too small** (too close), your boxes may be **too loose** → lower β median (e.g., 0.92) or increase `sigma_ln`.
* **Check class priors**: If paragliders consistently overestimated distance, the assumed `S` may be too **large** → reduce median.
* **CI coverage**: With a small labeled set, verify that true distances fall in 90% CI about 90% of the time.

---

## 10) Edge cases & guardrails

* **s\_px < s\_min\_px** → skip: probably noise/airliner/cloud speck.
* **No overlapping priors** with classes → throw explicit error (add missing prior or drop class).
* **Degenerate boxes** `(w=0 or h=0)` → handled (s\_px from other dimension or raises if zero).
* **Distance cap**: We do not drop results, but annotate with `note` to aid filtering upstream.

---

## 11) Deployment notes (Android)

* Compute `f_px` once per camera configuration change.
* Prefer `size_measure="geom_mean"` for rotated/elongated objects.
* Use fast math (`StrictMath` / intrinsics). Precompute `z` for CI (e.g., `z05≈-1.64485`, `z95≈1.64485`).
* Consider porting hot path to **C++** (JNI/NDK) if needed; the code maps 1:1 to scalar ops.

---

## 12) FAQ

**Q: Does this give ground range or slant range?**
A: **Slant range**. If you know camera tilt and want horizontal ground range, add a post-step using the object’s elevation angle.

**Q: How do we “ignore clouds and high planes”?**
A: They appear as **very small** boxes. Use `s_min_px` and/or drop any result with `distance_median` above your operational ceiling.

**Q: What if class probabilities are wrong?**
A: The mixture still works; the CI will typically widen. You can clamp to the top-k classes or renormalize after pruning unlikely classes.

**Q: Can we use segmentation masks?**
A: Yes. Replace `s_px` by an effective diameter from mask area: `s_eff = 2*sqrt(A/pi)`; reduce size-noise accordingly.

---

## 13) Mathematical appendix

* **Per-class lognormal parameters:**
  $\mu_D = \ln f_{px} + \mu_{S,c} - \mu_\beta - \ln \bar{s}$, $\sigma_D^2 = \sigma_{S,c}^2 + \sigma_\beta^2 + \sigma_s^2$.
* **Per-class stats:**
  mode = $\exp(\mu_D - \sigma_D^2)$, median = $\exp(\mu_D)$, mean = $\exp(\mu_D + \tfrac{1}{2}\sigma_D^2)$.
  Quantile $q$: $\exp(\mu_D + z_q\,\sigma_D)$, where $z_q$ is the normal quantile.
* **Mixture moment-match:**
  $m_1 = \sum_c p(c)\,\exp(\mu_D + \tfrac{1}{2}\sigma_D^2)$,
  $m_2 = \sum_c p(c)\,\exp(2\mu_D + 2\sigma_D^2)$.
  Equivalent lognormal: $V^* = \ln(m_2/m_1^2)$, $\mu^* = \ln m_1 - \tfrac{1}{2}V^*$.

---

## 14) Change log

* **1.0** – Initial release: core geometry, modular uncertainties, class mixture, moment-matched CI, mobile-friendly implementation.
