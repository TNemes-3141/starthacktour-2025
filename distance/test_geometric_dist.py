#!/usr/bin/env python3
"""
Comprehensive test suite for geometric_dist.py

Tests all features, edge cases, and mathematical correctness of the 
monocular distance estimator.
"""

import sys
import os
import math
import pytest
from typing import Dict, List, Any

# Add the distance module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometric_dist import (
    CameraParams, EstimatorConfig, ClassSizePriors, LogNormal,
    TightnessPrior, SizeNoiseModel, Heuristics,
    estimate_detection_distance, estimate_distances, default_priors,
    _normal_ppf, _size_from_box, _per_class_log_params
)


class TestNormalQuantile:
    """Test the normal quantile approximation function."""
    
    def test_normal_ppf_boundary_values(self):
        """Test boundary values for normal quantile function."""
        # Test exact boundary cases
        assert _normal_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
        
        # Test extreme values
        assert _normal_ppf(1e-15) == pytest.approx(-7.94, abs=0.1)  # Very negative
        assert _normal_ppf(1-1e-15) == pytest.approx(7.94, abs=0.1)  # Very positive
        
    def test_normal_ppf_known_values(self):
        """Test against known quantile values."""
        # Standard normal quantiles
        assert _normal_ppf(0.05) == pytest.approx(-1.64485, abs=1e-4)
        assert _normal_ppf(0.95) == pytest.approx(1.64485, abs=1e-4)
        assert _normal_ppf(0.025) == pytest.approx(-1.95996, abs=1e-4)
        assert _normal_ppf(0.975) == pytest.approx(1.95996, abs=1e-4)
        
    def test_normal_ppf_symmetry(self):
        """Test symmetry property of normal quantiles."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert _normal_ppf(p) == pytest.approx(-_normal_ppf(1-p), abs=1e-6)
            
    def test_normal_ppf_edge_cases(self):
        """Test edge cases that should raise errors."""
        with pytest.raises(ValueError):
            _normal_ppf(0.0)
        with pytest.raises(ValueError):
            _normal_ppf(1.0)
        with pytest.raises(ValueError):
            _normal_ppf(-0.1)
        with pytest.raises(ValueError):
            _normal_ppf(1.1)


class TestCameraParams:
    """Test camera parameter calculations."""
    
    def test_f_px_direct(self):
        """Test direct f_px specification."""
        cam = CameraParams(f_px=2750.0)
        assert cam.effective_f_px() == pytest.approx(2750.0)
        
    def test_hfov_calculation(self):
        """Test f_px calculation from HFOV."""
        cam = CameraParams(hfov_rad=math.radians(60.0), image_width_px=1000)
        expected_f_px = 1000 / (2.0 * math.tan(math.radians(30.0)))
        assert cam.effective_f_px() == pytest.approx(expected_f_px, rel=1e-6)
        
    def test_hfov_edge_cases(self):
        """Test edge cases for HFOV."""
        # Very narrow HFOV
        cam = CameraParams(hfov_rad=math.radians(1.0), image_width_px=1000)
        assert cam.effective_f_px() > 10000  # Should be very large
        
        # Very wide HFOV
        cam = CameraParams(hfov_rad=math.radians(170.0), image_width_px=1000)
        assert cam.effective_f_px() > 0  # Should be positive but small
        
    def test_camera_validation_errors(self):
        """Test validation errors for camera parameters."""
        # Missing both f_px and HFOV
        cam = CameraParams()
        with pytest.raises(ValueError, match="Provide either f_px"):
            cam.effective_f_px()
            
        # Missing image width
        cam = CameraParams(hfov_rad=math.radians(60.0))
        with pytest.raises(ValueError, match="Provide either f_px"):
            cam.effective_f_px()
            
        # Invalid HFOV
        cam = CameraParams(hfov_rad=0.0, image_width_px=1000)
        with pytest.raises(ValueError, match="hfov_rad must be in"):
            cam.effective_f_px()
            
        cam = CameraParams(hfov_rad=math.pi, image_width_px=1000)
        with pytest.raises(ValueError, match="hfov_rad must be in"):
            cam.effective_f_px()


class TestLogNormal:
    """Test LogNormal distribution class."""
    
    def test_from_median_sigma_ln(self):
        """Test LogNormal creation from median and sigma_ln."""
        ln = LogNormal.from_median_sigma_ln(10.0, 0.5)
        assert ln.mu_ln == pytest.approx(math.log(10.0))
        assert ln.sigma_ln == pytest.approx(0.5)
        
    def test_lognormal_quantiles(self):
        """Test quantile calculations."""
        ln = LogNormal.from_median_sigma_ln(10.0, 0.5)
        
        # Median should be exactly 10.0
        assert ln.quantile(0.5) == pytest.approx(10.0, rel=1e-6)
        
        # Lower and upper quantiles should be symmetric on log scale
        q05 = ln.quantile(0.05)
        q95 = ln.quantile(0.95)
        assert math.log(q95/10.0) == pytest.approx(-math.log(q05/10.0), rel=1e-4)
        
    def test_lognormal_invalid_median(self):
        """Test error for invalid median."""
        with pytest.raises(ValueError, match="median must be > 0"):
            LogNormal.from_median_sigma_ln(0.0, 0.5)
        with pytest.raises(ValueError, match="median must be > 0"):
            LogNormal.from_median_sigma_ln(-1.0, 0.5)


class TestSizeMeasures:
    """Test pixel size calculation from bounding boxes."""
    
    def test_size_from_box_geom_mean(self):
        """Test geometric mean size calculation."""
        assert _size_from_box(20, 30, "geom_mean") == pytest.approx(math.sqrt(600), rel=1e-6)
        assert _size_from_box(0, 30, "geom_mean") == 0.0
        assert _size_from_box(20, 0, "geom_mean") == 0.0
        
    def test_size_from_box_max(self):
        """Test max size calculation."""
        assert _size_from_box(20, 30, "max") == 30.0
        assert _size_from_box(30, 20, "max") == 30.0
        assert _size_from_box(0, 30, "max") == 30.0
        
    def test_size_from_box_mean(self):
        """Test mean size calculation."""
        assert _size_from_box(20, 30, "mean") == 25.0
        assert _size_from_box(0, 30, "mean") == 15.0
        
    def test_size_from_box_negative_values(self):
        """Test handling of negative input values."""
        # Should convert to zero
        assert _size_from_box(-10, 20, "geom_mean") == 0.0
        assert _size_from_box(20, -10, "geom_mean") == 0.0
        assert _size_from_box(-10, 20, "max") == 20.0


class TestPerClassLogParams:
    """Test per-class lognormal parameter computation."""
    
    def test_per_class_log_params_basic(self):
        """Test basic per-class parameter calculation."""
        size_prior = LogNormal(mu_ln=math.log(5.0), sigma_ln=0.3)
        mu, sigma = _per_class_log_params(
            f_px=1000.0,
            s_px=20.0,
            size_prior=size_prior,
            beta_mu_ln=math.log(0.95),
            beta_sigma_ln=0.05,
            s_sigma_ln=0.1
        )
        
        # Check that mu follows the expected formula
        expected_mu = (math.log(1000.0) + math.log(5.0) - 
                      math.log(0.95) - math.log(20.0))
        assert mu == pytest.approx(expected_mu, rel=1e-6)
        
        # Check that sigma combines variances correctly
        expected_var = 0.3**2 + 0.05**2 + 0.1**2
        assert sigma**2 == pytest.approx(expected_var, rel=1e-6)
        
    def test_per_class_log_params_zero_s_px(self):
        """Test error handling for zero s_px."""
        size_prior = LogNormal(mu_ln=0.0, sigma_ln=0.3)
        with pytest.raises(ValueError, match="s_px must be > 0"):
            _per_class_log_params(
                f_px=1000.0,
                s_px=0.0,
                size_prior=size_prior,
                beta_mu_ln=0.0,
                beta_sigma_ln=0.0,
                s_sigma_ln=0.0
            )


class TestSizeNoiseModel:
    """Test size noise model calculations."""
    
    def test_size_noise_disabled(self):
        """Test size noise when disabled."""
        model = SizeNoiseModel(enabled=False)
        assert model.sigma_ln_for_box(100, 200) == 0.0
        
    def test_size_noise_fixed_sigma(self):
        """Test fixed sigma_ln."""
        model = SizeNoiseModel(enabled=True, fixed_sigma_ln=0.08)
        assert model.sigma_ln_for_box(100, 200) == 0.08
        assert model.sigma_ln_for_box(10, 10) == 0.08
        
    def test_size_noise_dynamic(self):
        """Test dynamic sigma_ln calculation."""
        model = SizeNoiseModel(enabled=True, fixed_sigma_ln=None, c0=2.0, cap=0.15)
        
        # Large box -> low noise
        sigma_large = model.sigma_ln_for_box(1000, 1000)
        expected_large = min(0.15, 2.0 / math.sqrt(1000000))
        assert sigma_large == pytest.approx(expected_large, rel=1e-6)
        
        # Small box -> high noise (capped)
        sigma_small = model.sigma_ln_for_box(1, 1)
        expected_small = min(0.15, 2.0 / math.sqrt(1))
        assert sigma_small == pytest.approx(expected_small, rel=1e-6)
        assert sigma_small == 0.15  # Should be capped
        
    def test_size_noise_edge_cases(self):
        """Test edge cases for size noise."""
        model = SizeNoiseModel(enabled=True, fixed_sigma_ln=None, c0=2.0, cap=0.15)
        
        # Zero area (should use minimum area = 1.0)
        sigma_zero = model.sigma_ln_for_box(0, 0)
        expected_zero = min(0.15, 2.0 / math.sqrt(1.0))
        assert sigma_zero == pytest.approx(expected_zero, rel=1e-6)


class TestClassSizePriors:
    """Test class size priors management."""
    
    def test_class_size_priors_get(self):
        """Test getting priors for existing class."""
        priors = ClassSizePriors({
            "bird": LogNormal.from_median_sigma_ln(0.5, 0.4),
            "drone": LogNormal.from_median_sigma_ln(0.3, 0.25)
        })
        
        bird_prior = priors.get("bird")
        assert bird_prior.mu_ln == pytest.approx(math.log(0.5))
        assert bird_prior.sigma_ln == pytest.approx(0.4)
        
    def test_class_size_priors_missing(self):
        """Test error for missing class."""
        priors = ClassSizePriors({"bird": LogNormal.from_median_sigma_ln(0.5, 0.4)})
        
        with pytest.raises(KeyError, match="Missing size prior for class 'drone'"):
            priors.get("drone")


class TestTightnessPrior:
    """Test tightness prior configuration."""
    
    def test_tightness_enabled(self):
        """Test tightness prior when enabled."""
        prior = TightnessPrior(enabled=True, mu_ln=math.log(0.95), sigma_ln=0.05)
        mu, sigma = prior.params()
        assert mu == pytest.approx(math.log(0.95))
        assert sigma == pytest.approx(0.05)
        
    def test_tightness_disabled(self):
        """Test tightness prior when disabled."""
        prior = TightnessPrior(enabled=False, mu_ln=math.log(0.95), sigma_ln=0.05)
        mu, sigma = prior.params()
        assert mu == 0.0
        assert sigma == 0.0


class TestEstimatorConfig:
    """Test estimator configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        cfg = EstimatorConfig()
        assert cfg.size_measure == "geom_mean"
        assert cfg.ci_lower == 0.05
        assert cfg.ci_upper == 0.95
        assert cfg.tightness_prior.enabled is True
        assert cfg.size_noise.enabled is True
        assert cfg.heuristics.s_min_px == 6.0


class TestDetectionInputValidation:
    """Test detection input validation and parsing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "bird": LogNormal.from_median_sigma_ln(0.5, 0.4),
                "drone": LogNormal.from_median_sigma_ln(0.3, 0.25)
            })
        )
        
    def test_detection_bbox_input(self):
        """Test detection with bbox input."""
        detection = {
            "bbox": [100, 100, 120, 110],
            "class_probs": {"bird": 0.7, "drone": 0.3}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        assert "distance_m_mode" in result
        assert "distance_m_median" in result
        
    def test_detection_wh_input(self):
        """Test detection with w,h input."""
        detection = {
            "w": 20,
            "h": 10,
            "class_probs": {"bird": 0.7, "drone": 0.3}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        assert "distance_m_mode" in result
        assert "distance_m_median" in result
        
    def test_detection_size_px_override(self):
        """Test detection with explicit size_px."""
        detection = {
            "bbox": [100, 100, 120, 110],
            "size_px": 25.0,
            "class_probs": {"bird": 0.7, "drone": 0.3}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        assert "distance_m_mode" in result
        
    def test_detection_missing_dimensions(self):
        """Test detection missing both bbox and w,h."""
        detection = {
            "class_probs": {"bird": 0.7, "drone": 0.3}
        }
        with pytest.raises(KeyError, match="Detection must include"):
            estimate_detection_distance(detection, self.cam, self.cfg)
            
    def test_detection_missing_class_probs(self):
        """Test detection missing class probabilities."""
        detection = {
            "bbox": [100, 100, 120, 110]
        }
        with pytest.raises(KeyError, match="Detection missing class_probs"):
            estimate_detection_distance(detection, self.cam, self.cfg)
            
    def test_detection_no_matching_classes(self):
        """Test detection with no classes that have priors."""
        detection = {
            "bbox": [100, 100, 120, 110],
            "class_probs": {"airplane": 0.8, "helicopter": 0.2}
        }
        with pytest.raises(KeyError, match="No overlapping classes"):
            estimate_detection_distance(detection, self.cam, self.cfg)
            
    def test_detection_zero_class_probs(self):
        """Test detection with all zero class probabilities."""
        detection = {
            "bbox": [100, 100, 120, 110],
            "class_probs": {"bird": 0.0, "drone": 0.0}
        }
        with pytest.raises(ValueError, match="class_probs must have positive mass"):
            estimate_detection_distance(detection, self.cam, self.cfg)


class TestTinyObjectGating:
    """Test tiny object filtering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "bird": LogNormal.from_median_sigma_ln(0.5, 0.4)
            }),
            heuristics=Heuristics(s_min_px=6.0)
        )
        
    def test_tiny_object_skipped(self):
        """Test that tiny objects are skipped."""
        detection = {
            "w": 3,  # Less than s_min_px = 6
            "h": 2,
            "class_probs": {"bird": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        assert result["distance_m_mode"] is None
        assert result["distance_m_median"] is None
        assert result["distance_m_mean"] is None
        assert result["distance_m_ci90"] is None
        assert "skipped" in result["note"]
        
    def test_large_enough_object_processed(self):
        """Test that objects above threshold are processed."""
        detection = {
            "w": 10,  # Greater than s_min_px = 6
            "h": 8,
            "class_probs": {"bird": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        assert result["distance_m_mode"] is not None
        assert result["distance_m_median"] is not None
        assert result["distance_m_mean"] is not None
        assert result["distance_m_ci90"] is not None


class TestDistanceCapAnnotation:
    """Test distance cap annotation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        # Use a large size prior to trigger distance cap
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "large_object": LogNormal.from_median_sigma_ln(100.0, 0.2)  # Very large
            }),
            heuristics=Heuristics(distance_cap_m=500.0)
        )
        
    def test_distance_cap_annotation(self):
        """Test that distance cap generates annotation."""
        detection = {
            "w": 10,  # Small pixel size + large object = far distance
            "h": 10,
            "class_probs": {"large_object": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # Should have distance estimates but also a warning note
        assert result["distance_m_median"] is not None
        if result["distance_m_median"] > 500.0:
            assert "note" in result
            assert "distance_median>" in result["note"]


class TestMathematicalCorrectness:
    """Test mathematical correctness of calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "test_object": LogNormal.from_median_sigma_ln(2.0, 0.1)  # Low uncertainty
            }),
            tightness_prior=TightnessPrior(enabled=False),  # Disable for simplicity
            size_noise=SizeNoiseModel(enabled=False)  # Disable for simplicity
        )
        
    def test_distance_formula_basic(self):
        """Test basic distance formula: D = f_px * S / s_px."""
        detection = {
            "w": 100,
            "h": 100,  # s_px = sqrt(10000) = 100
            "class_probs": {"test_object": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # With f_px=1000, S≈2.0, s_px=100: D ≈ 1000*2/100 = 20
        expected_distance = 1000.0 * 2.0 / 100.0
        assert result["distance_m_median"] == pytest.approx(expected_distance, rel=0.1)
        
    def test_distance_scaling_with_pixel_size(self):
        """Test that distance scales inversely with pixel size."""
        base_detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"test_object": 1.0}
        }
        base_result = estimate_detection_distance(base_detection, self.cam, self.cfg)
        
        # Half the pixel size should double the distance
        small_detection = {
            "w": 50,
            "h": 50,
            "class_probs": {"test_object": 1.0}
        }
        small_result = estimate_detection_distance(small_detection, self.cam, self.cfg)
        
        ratio = small_result["distance_m_median"] / base_result["distance_m_median"]
        assert ratio == pytest.approx(2.0, rel=0.1)
        
    def test_lognormal_statistics(self):
        """Test lognormal statistics are correct."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"test_object": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # For lognormal: mean > median > mode (when sigma > 0)
        mode = result["distance_m_mode"]
        median = result["distance_m_median"]
        mean = result["distance_m_mean"]
        
        # With low uncertainty, they should be close but follow the order
        assert mode <= median <= mean
        
    def test_confidence_interval_contains_median(self):
        """Test that confidence interval contains the median."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"test_object": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        median = result["distance_m_median"]
        ci_lo, ci_hi = result["distance_m_ci90"]
        
        assert ci_lo <= median <= ci_hi


class TestClassMixtures:
    """Test class probability mixtures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "small": LogNormal.from_median_sigma_ln(1.0, 0.1),
                "large": LogNormal.from_median_sigma_ln(10.0, 0.1)
            }),
            tightness_prior=TightnessPrior(enabled=False),
            size_noise=SizeNoiseModel(enabled=False)
        )
        
    def test_single_class_mixture(self):
        """Test mixture with single class."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"small": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # Should have per-class results
        assert len(result["per_class"]) == 1
        assert result["per_class"][0]["class"] == "small"
        assert result["per_class"][0]["p"] == 1.0
        
        # Mode should come from the single class
        assert result["distance_m_mode"] == pytest.approx(
            result["per_class"][0]["mode"], rel=1e-6)
            
    def test_two_class_mixture(self):
        """Test mixture with two classes."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"small": 0.3, "large": 0.7}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # Should have two per-class results
        assert len(result["per_class"]) == 2
        
        # Class probabilities should be normalized
        probs = [cls["p"] for cls in result["per_class"]]
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        
        # Mode should come from the highest probability class (large)
        large_class = next(cls for cls in result["per_class"] if cls["class"] == "large")
        assert result["distance_m_mode"] == pytest.approx(large_class["mode"], rel=1e-6)
        
    def test_mixture_probability_normalization(self):
        """Test that class probabilities are normalized."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"small": 2.0, "large": 8.0}  # Sum = 10
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # Should be normalized to 0.2 and 0.8
        probs = {cls["class"]: cls["p"] for cls in result["per_class"]}
        assert probs["small"] == pytest.approx(0.2, abs=1e-6)
        assert probs["large"] == pytest.approx(0.8, abs=1e-6)


class TestUncertaintyModules:
    """Test individual uncertainty modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.base_cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "test": LogNormal.from_median_sigma_ln(2.0, 0.1)
            })
        )
        
    def test_tightness_prior_effect(self):
        """Test effect of tightness prior."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"test": 1.0}
        }
        
        # With tightness enabled
        cfg_with_tightness = EstimatorConfig(
            class_size_priors=self.base_cfg.class_size_priors,
            tightness_prior=TightnessPrior(enabled=True, mu_ln=math.log(0.9), sigma_ln=0.1),
            size_noise=SizeNoiseModel(enabled=False)
        )
        result_with = estimate_detection_distance(detection, self.cam, cfg_with_tightness)
        
        # Without tightness
        cfg_without_tightness = EstimatorConfig(
            class_size_priors=self.base_cfg.class_size_priors,
            tightness_prior=TightnessPrior(enabled=False),
            size_noise=SizeNoiseModel(enabled=False)
        )
        result_without = estimate_detection_distance(detection, self.cam, cfg_without_tightness)
        
        # With tightness (β < 1), effective pixel size is smaller, so distance should be larger
        assert result_with["distance_m_median"] > result_without["distance_m_median"]
        
    def test_size_noise_effect(self):
        """Test effect of size noise."""
        detection = {
            "w": 100,
            "h": 100,
            "class_probs": {"test": 1.0}
        }
        
        # With size noise
        cfg_with_noise = EstimatorConfig(
            class_size_priors=self.base_cfg.class_size_priors,
            tightness_prior=TightnessPrior(enabled=False),
            size_noise=SizeNoiseModel(enabled=True, fixed_sigma_ln=0.2)
        )
        result_with = estimate_detection_distance(detection, self.cam, cfg_with_noise)
        
        # Without size noise
        cfg_without_noise = EstimatorConfig(
            class_size_priors=self.base_cfg.class_size_priors,
            tightness_prior=TightnessPrior(enabled=False),
            size_noise=SizeNoiseModel(enabled=False)
        )
        result_without = estimate_detection_distance(detection, self.cam, cfg_without_noise)
        
        # With noise, confidence interval should be wider
        with_width = result_with["distance_m_ci90"][1] - result_with["distance_m_ci90"][0]
        without_width = result_without["distance_m_ci90"][1] - result_without["distance_m_ci90"][0]
        
        assert with_width > without_width


class TestVectorizedInterface:
    """Test the vectorized interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "bird": LogNormal.from_median_sigma_ln(0.5, 0.4),
                "drone": LogNormal.from_median_sigma_ln(0.3, 0.25)
            })
        )
        
    def test_estimate_distances_multiple(self):
        """Test estimating distances for multiple detections."""
        detections = [
            {"w": 20, "h": 15, "class_probs": {"bird": 0.8, "drone": 0.2}},
            {"w": 30, "h": 25, "class_probs": {"drone": 1.0}},
            {"w": 100, "h": 80, "class_probs": {"bird": 0.6, "drone": 0.4}}
        ]
        
        results = estimate_distances(detections, self.cam, self.cfg)
        
        assert len(results) == 3
        for result in results:
            assert "distance_m_mode" in result
            assert "distance_m_median" in result
            assert "distance_m_mean" in result
            assert "distance_m_ci90" in result
            
    def test_estimate_distances_error_handling(self):
        """Test error handling in vectorized interface."""
        detections = [
            {"w": 20, "h": 15, "class_probs": {"bird": 0.8}},  # Valid
            {"class_probs": {"bird": 0.8}},  # Missing dimensions
            {"w": 30, "h": 25, "class_probs": {"airplane": 1.0}}  # Unknown class
        ]
        
        results = estimate_distances(detections, self.cam, self.cfg)
        
        assert len(results) == 3
        assert "distance_m_mode" in results[0]  # First should succeed
        assert "error" in results[1]  # Second should have error
        assert "error" in results[2]  # Third should have error


class TestDefaultPriors:
    """Test the default priors function."""
    
    def test_default_priors_structure(self):
        """Test that default priors have expected structure."""
        priors = default_priors()
        
        expected_classes = {
            "bird", "drone", "paraglider", "balloon", "helicopter",
            "light_ac", "kite", "rc_plane", "parachute"
        }
        
        assert set(priors.priors.keys()) == expected_classes
        
        # All should be LogNormal instances
        for cls, prior in priors.priors.items():
            assert isinstance(prior, LogNormal)
            assert prior.sigma_ln > 0  # Should have positive uncertainty
            
    def test_default_priors_reasonable_values(self):
        """Test that default priors have reasonable median sizes."""
        priors = default_priors()
        
        # Birds should be smaller than helicopters
        bird_median = math.exp(priors.priors["bird"].mu_ln)
        helicopter_median = math.exp(priors.priors["helicopter"].mu_ln)
        assert bird_median < helicopter_median
        
        # Drones should be smaller than paragliders
        drone_median = math.exp(priors.priors["drone"].mu_ln)
        paraglider_median = math.exp(priors.priors["paraglider"].mu_ln)
        assert drone_median < paraglider_median


class TestOutputSchema:
    """Test the output schema matches documentation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cam = CameraParams(f_px=1000.0)
        self.cfg = EstimatorConfig(
            class_size_priors=ClassSizePriors({
                "bird": LogNormal.from_median_sigma_ln(0.5, 0.4)
            })
        )
        
    def test_output_schema_completeness(self):
        """Test that output contains all expected fields."""
        detection = {
            "w": 20,
            "h": 15,
            "class_probs": {"bird": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # Core distance estimates
        assert "distance_m_mode" in result
        assert "distance_m_median" in result
        assert "distance_m_mean" in result
        assert "distance_m_ci90" in result
        
        # Per-class breakdown
        assert "per_class" in result
        assert len(result["per_class"]) == 1
        
        per_class = result["per_class"][0]
        assert "class" in per_class
        assert "p" in per_class
        assert "mode" in per_class
        assert "median" in per_class
        assert "mean" in per_class
        assert "ci" in per_class
        assert "ln_params" in per_class
        
        # Parameters used
        assert "params_used" in result
        params = result["params_used"]
        assert "f_px" in params
        assert "size_measure" in params
        assert "beta" in params
        assert "s_noise" in params
        assert "ci" in params
        
        # Original detection fields should be preserved
        assert "w" in result
        assert "h" in result
        assert "class_probs" in result
        
    def test_output_types(self):
        """Test that output values have correct types."""
        detection = {
            "w": 20,
            "h": 15,
            "class_probs": {"bird": 1.0}
        }
        result = estimate_detection_distance(detection, self.cam, self.cfg)
        
        # All distance values should be floats
        assert isinstance(result["distance_m_mode"], float)
        assert isinstance(result["distance_m_median"], float)
        assert isinstance(result["distance_m_mean"], float)
        
        # CI should be list of two floats
        ci = result["distance_m_ci90"]
        assert isinstance(ci, list)
        assert len(ci) == 2
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)
        assert ci[0] < ci[1]  # Lower bound < upper bound


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
