"""Comprehensive tests for mathematical validation and Shannon entropy calculation.

This module provides property-based testing for mathematical algorithms and
validation functions to ensure correctness under all edge cases.
"""

import math
import pytest
from collections import Counter
from hypothesis import given, strategies as st, assume, settings
from typing import Any

from src.utils.validation import RobustScoreValidator, MathematicalValidator


class TestRobustScoreValidator:
    """Tests for robust score validation with edge case handling."""

    def test_valid_probability_scores(self):
        """Test validation of valid probability scores."""
        test_cases = [0.0, 0.5, 1.0, 0.25, 0.75, 0.999]

        for score in test_cases:
            result = RobustScoreValidator.validate_probability_score(score)
            assert result == score, f"Valid score {score} was modified"
            assert 0.0 <= result <= 1.0

    def test_nan_handling(self):
        """Test NaN value handling."""
        result = RobustScoreValidator.validate_probability_score(
            value=float('nan'),
            field_name="test_score",
            default_value=0.5
        )
        assert result == 0.5
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_infinity_handling(self):
        """Test infinity value handling."""
        # Positive infinity
        result_pos = RobustScoreValidator.validate_probability_score(
            value=float('inf'),
            field_name="test_score"
        )
        assert result_pos == 1.0

        # Negative infinity
        result_neg = RobustScoreValidator.validate_probability_score(
            value=float('-inf'),
            field_name="test_score"
        )
        assert result_neg == 0.0

    def test_out_of_range_values(self):
        """Test handling of out-of-range values."""
        test_cases = [
            (-0.5, 0.0),
            (-1.0, 0.0),
            (1.5, 1.0),
            (2.0, 1.0),
            (100.0, 1.0),
            (-100.0, 0.0)
        ]

        for input_val, expected in test_cases:
            result = RobustScoreValidator.validate_probability_score(input_val)
            assert result == expected, f"Input {input_val} should give {expected}, got {result}"

    def test_none_handling(self):
        """Test None value handling."""
        result = RobustScoreValidator.validate_probability_score(
            value=None,
            field_name="test_score",
            default_value=0.3
        )
        assert result == 0.3

    @given(st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.none(),
        st.text(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers())
    ))
    def test_validation_robustness(self, invalid_input):
        """Property-based test: validator handles any input gracefully."""
        result = RobustScoreValidator.validate_probability_score(
            value=invalid_input,
            field_name="test_field",
            default_value=0.3
        )

        # Should always return a valid float
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_positive_score_validation(self):
        """Test positive score validation with custom ranges."""
        # Valid cases
        assert RobustScoreValidator.validate_positive_score(5.0, max_value=10.0) == 5.0
        assert RobustScoreValidator.validate_positive_score(0.0) == 0.0

        # Clamping cases
        assert RobustScoreValidator.validate_positive_score(15.0, max_value=10.0) == 10.0
        assert RobustScoreValidator.validate_positive_score(-5.0) == 0.0

        # Edge cases
        assert RobustScoreValidator.validate_positive_score(float('inf'), max_value=5.0) == 5.0
        assert RobustScoreValidator.validate_positive_score(float('nan'), default_value=2.0) == 2.0


class TestShannonEntropyCalculation:
    """Comprehensive tests for Shannon entropy calculation with mathematical properties."""

    def test_empty_domains(self):
        """Test edge case: empty domain list."""
        result = MathematicalValidator.calculate_normalized_shannon_entropy([])
        assert result == 0.0

    def test_single_domain(self):
        """Test edge case: single domain (no diversity)."""
        assert MathematicalValidator.calculate_normalized_shannon_entropy(["domain1"]) == 0.0
        assert MathematicalValidator.calculate_normalized_shannon_entropy(["domain1"] * 5) == 0.0

    def test_perfect_diversity(self):
        """Test perfect diversity cases (should return 1.0)."""
        # Two different domains
        result = MathematicalValidator.calculate_normalized_shannon_entropy(["domain1", "domain2"])
        assert abs(result - 1.0) < 1e-10

        # Three different domains
        result = MathematicalValidator.calculate_normalized_shannon_entropy(
            ["domain1", "domain2", "domain3"]
        )
        assert abs(result - 1.0) < 1e-10

        # Many different domains
        domains = [f"domain{i}" for i in range(10)]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(domains)
        assert abs(result - 1.0) < 1e-10

    def test_known_entropy_values(self):
        """Test with known mathematical values."""
        # Half and half distribution should give entropy close to 1.0
        domains = ["domain1", "domain1", "domain2", "domain2"]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(domains)
        assert abs(result - 1.0) < 1e-10

        # Uneven distribution: 3:1 ratio with 2 domains
        domains = ["domain1"] * 3 + ["domain2"]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(domains)

        # Manually calculated: entropy = -(0.75*log2(0.75) + 0.25*log2(0.25)) / log2(2)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)) / math.log2(2)
        assert abs(result - expected) < 1e-10

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=100))
    def test_entropy_properties_hypothesis(self, domains):
        """Property-based testing using Hypothesis."""
        # Filter out empty strings
        domains = [d for d in domains if d.strip()]
        assume(len(domains) > 0)  # Ensure we have valid domains

        result = MathematicalValidator.calculate_normalized_shannon_entropy(domains)

        # Entropy should always be between 0 and 1
        assert 0.0 <= result <= 1.0

        # If all domains are the same, entropy should be 0
        if len(set(domains)) == 1:
            assert result == 0.0

        # If all domains are different, entropy should be 1.0
        if len(set(domains)) == len(domains) and len(domains) > 1:
            assert abs(result - 1.0) < 1e-10

    @given(st.integers(min_value=2, max_value=20))
    def test_max_diversity_scales(self, num_domains):
        """Test that max diversity is always 1.0 regardless of number of domains."""
        domains = [f"domain{i}" for i in range(num_domains)]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(domains)
        assert abs(result - 1.0) < 1e-10

    def test_entropy_monotonicity(self):
        """Test entropy behavior with increasing diversity."""
        # Start with uniform distribution
        base_domains = ["domain1", "domain2"]
        base_entropy = MathematicalValidator.calculate_normalized_shannon_entropy(base_domains)

        # Perfect diversity should always be 1.0
        assert abs(base_entropy - 1.0) < 1e-10

        # Test with different distribution patterns
        skewed_domains = ["domain1"] * 3 + ["domain2"]
        skewed_entropy = MathematicalValidator.calculate_normalized_shannon_entropy(skewed_domains)

        # Skewed distribution should have lower entropy than perfect distribution
        assert skewed_entropy < 1.0

    def test_mathematical_invariants(self):
        """Test mathematical invariants of Shannon entropy."""
        # Test commutativity: order doesn't matter
        domains1 = ["a", "b", "a", "c", "b"]
        domains2 = ["b", "a", "c", "a", "b"]

        entropy1 = MathematicalValidator.calculate_normalized_shannon_entropy(domains1)
        entropy2 = MathematicalValidator.calculate_normalized_shannon_entropy(domains2)
        assert abs(entropy1 - entropy2) < 1e-10

        # Test that adding more of existing domains decreases entropy
        more_skewed = domains1 + ["a", "a"]
        entropy_skewed = MathematicalValidator.calculate_normalized_shannon_entropy(more_skewed)
        assert entropy_skewed <= entropy1

    def test_input_validation(self):
        """Test input validation for entropy calculation."""
        # Test invalid inputs
        with pytest.raises(ValueError):
            MathematicalValidator.calculate_normalized_shannon_entropy("not a list")

        with pytest.raises(ValueError):
            MathematicalValidator.calculate_normalized_shannon_entropy([1, 2, 3])  # Not strings

        with pytest.raises(ValueError):
            MathematicalValidator.calculate_normalized_shannon_entropy(["", "valid"])  # Empty string

    def test_edge_case_domains(self):
        """Test edge cases with domain strings."""
        # Very long domain names
        long_domains = ["a" * 1000, "b" * 1000]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(long_domains)
        assert abs(result - 1.0) < 1e-10

        # Special characters
        special_domains = ["domain-1", "domain_2", "domain.3"]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(special_domains)
        assert abs(result - 1.0) < 1e-10

        # Unicode characters
        unicode_domains = ["域名1", "डोमेन2", "домен3"]
        result = MathematicalValidator.calculate_normalized_shannon_entropy(unicode_domains)
        assert abs(result - 1.0) < 1e-10


class TestQualityScoreValidation:
    """Tests for quality score component validation."""

    def test_valid_components(self):
        """Test validation with valid components."""
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=0.8,
            convergence_rate=0.6,
            contradiction_rate=0.2,
            pattern_strength=0.9,
            confidence_avg=0.75
        )

        assert components['diversity'] == 0.8
        assert components['convergence'] == 0.6
        assert components['contradiction'] == 0.2
        assert components['pattern_strength'] == 0.9
        assert components['confidence'] == 0.75

    def test_edge_case_validation(self):
        """Test validation with edge cases."""
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=float('nan'),
            convergence_rate=float('inf'),
            contradiction_rate=-0.5,
            pattern_strength=1.5,
            confidence_avg=None
        )

        assert 0.0 <= components['diversity'] <= 1.0
        assert 0.0 <= components['convergence'] <= 1.0
        assert 0.0 <= components['contradiction'] <= 1.0
        assert 0.0 <= components['pattern_strength'] <= 1.0
        assert 'confidence' not in components  # Should be excluded when None

    def test_optional_components(self):
        """Test validation with optional components."""
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=0.5,
            convergence_rate=0.7,
            contradiction_rate=0.1
        )

        assert 'pattern_strength' not in components
        assert 'confidence' not in components
        assert len(components) == 3

    @given(
        diversity=st.one_of(
            st.floats(min_value=-10.0, max_value=10.0),
            st.just(float('nan')),
            st.just(float('inf')),
            st.just(float('-inf'))
        ),
        convergence=st.one_of(
            st.floats(min_value=-10.0, max_value=10.0),
            st.just(float('nan')),
            st.just(float('inf')),
            st.just(float('-inf'))
        ),
        contradiction=st.one_of(
            st.floats(min_value=-10.0, max_value=10.0),
            st.just(float('nan')),
            st.just(float('inf')),
            st.just(float('-inf'))
        )
    )
    def test_component_validation_properties(self, diversity, convergence, contradiction):
        """Property-based test for component validation."""
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=diversity,
            convergence_rate=convergence,
            contradiction_rate=contradiction
        )

        # All returned values should be valid probabilities
        for key, value in components.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
            assert not math.isnan(value)
            assert not math.isinf(value)


class TestIntegrationMathematicalValidation:
    """Integration tests combining multiple validation components."""

    def test_complete_validation_pipeline(self):
        """Test complete validation pipeline with realistic data."""
        # Simulate realistic research data
        domains = ["academic.edu", "research.org", "academic.edu", "news.com", "blog.net"]

        # Calculate entropy
        diversity = MathematicalValidator.calculate_normalized_shannon_entropy(domains)
        assert 0.0 <= diversity <= 1.0

        # Validate all components
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=diversity,
            convergence_rate=0.7,
            contradiction_rate=0.1,
            pattern_strength=0.8,
            confidence_avg=0.75
        )

        # All components should be valid
        for value in components.values():
            assert 0.0 <= value <= 1.0

    def test_stress_test_large_datasets(self):
        """Stress test with large datasets."""
        # Large domain list
        large_domains = [f"domain{i % 50}" for i in range(10000)]

        # Should complete without errors
        diversity = MathematicalValidator.calculate_normalized_shannon_entropy(large_domains)
        assert 0.0 <= diversity <= 1.0

        # Validation should handle large scores
        components = MathematicalValidator.validate_quality_score_components(
            diversity_score=diversity,
            convergence_rate=0.95,
            contradiction_rate=0.05
        )

        assert all(0.0 <= v <= 1.0 for v in components.values())

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small differences
        domains1 = ["a"] * 1000000 + ["b"]
        domains2 = ["a"] * 1000001 + ["b"]

        entropy1 = MathematicalValidator.calculate_normalized_shannon_entropy(domains1)
        entropy2 = MathematicalValidator.calculate_normalized_shannon_entropy(domains2)

        # Should be close but stable
        assert 0.0 <= entropy1 <= 1.0
        assert 0.0 <= entropy2 <= 1.0
        assert abs(entropy1 - entropy2) < 0.1  # Should be relatively close


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure for debugging
    ])
