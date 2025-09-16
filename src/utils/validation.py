"""Robust validation utilities for research data with comprehensive error handling.

This module provides validation functions that handle edge cases like NaN, infinity,
and type conversion errors gracefully while maintaining data integrity.
"""

import math
from typing import Any, TypedDict

import logfire


class SeverityResult(TypedDict):
    """Type definition for contradiction severity calculation result."""

    overall_severity: float
    raw_severity: float
    components: dict[str, float]
    metadata: dict[str, str | float]


class RobustScoreValidator:
    """Robust validation utilities for score fields with comprehensive error handling."""

    @staticmethod
    def validate_probability_score(
        value: Any, field_name: str = "score", default_value: float = 0.0, allow_none: bool = False
    ) -> float:
        """
        Robustly validate and clean probability scores (0.0 to 1.0).

        Handles edge cases:
        - NaN values -> default_value
        - Infinity values -> 1.0 (positive) or 0.0 (negative)
        - Out of range values -> clamped to [0.0, 1.0]
        - Type conversion errors -> default_value
        - None values -> default_value (with optional allowance)

        Args:
            value: Input value to validate
            field_name: Name of the field for logging purposes
            default_value: Default value for invalid inputs
            allow_none: Whether None values are acceptable

        Returns:
            Valid score between 0.0 and 1.0

        Examples:
            >>> RobustScoreValidator.validate_probability_score(0.8)
            0.8
            >>> RobustScoreValidator.validate_probability_score(float('nan'))
            0.0
            >>> RobustScoreValidator.validate_probability_score(1.5)
            1.0
            >>> RobustScoreValidator.validate_probability_score(-0.3)
            0.0
        """
        # Handle None values
        if value is None:
            if allow_none:
                return default_value
            logfire.warning(f"None value for {field_name}, using default {default_value}")
            return default_value

        # Convert to float if possible
        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            logfire.error(f"Cannot convert {field_name} value {value} to float: {e}")
            return default_value

        # Handle NaN
        if math.isnan(float_value):
            logfire.warning(f"NaN value detected for {field_name}, using default {default_value}")
            return default_value

        # Handle infinity
        if math.isinf(float_value):
            logfire.warning(f"Infinite value detected for {field_name}, using boundary value")
            return 1.0 if float_value > 0 else 0.0

        # Clamp to valid range [0.0, 1.0]
        clamped_value = max(0.0, min(1.0, float_value))

        # Log if clamping occurred
        if clamped_value != float_value:
            logfire.info(f"Clamped {field_name} from {float_value} to {clamped_value}")

        return clamped_value

    @staticmethod
    def validate_positive_score(
        value: Any, field_name: str = "score", max_value: float = 10.0, default_value: float = 1.0
    ) -> float:
        """
        Validate positive scores with configurable maximum.

        Args:
            value: Input value to validate
            field_name: Name of the field for logging
            max_value: Maximum allowed value
            default_value: Default value for invalid inputs

        Returns:
            Valid positive score between 0.0 and max_value

        Examples:
            >>> RobustScoreValidator.validate_positive_score(5.2, max_value=10.0)
            5.2
            >>> RobustScoreValidator.validate_positive_score(-1.0)
            0.0
            >>> RobustScoreValidator.validate_positive_score(float('inf'), max_value=5.0)
            5.0
        """
        if value is None:
            logfire.warning(f"None value for {field_name}, using default {default_value}")
            return default_value

        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            logfire.error(f"Cannot convert {field_name} value {value} to float: {e}")
            return default_value

        if math.isnan(float_value):
            logfire.warning(f"NaN value detected for {field_name}, using default {default_value}")
            return default_value

        if math.isinf(float_value):
            logfire.warning(f"Infinite value detected for {field_name}, using max value")
            return max_value if float_value > 0 else default_value

        # Ensure positive and within bounds
        clamped_value = max(0.0, min(max_value, float_value))

        if clamped_value != float_value:
            logfire.info(f"Clamped {field_name} from {float_value} to {clamped_value}")

        return clamped_value

    @staticmethod
    def validate_non_empty_string(
        value: Any, field_name: str = "field", default_value: str = ""
    ) -> str:
        """
        Validate that string fields are not empty and handle type conversion.

        Args:
            value: Input value to validate
            field_name: Name of the field for logging
            default_value: Default value for invalid inputs

        Returns:
            Valid non-empty string

        Raises:
            ValueError: If value cannot be converted to string or is empty
        """
        if value is None:
            if default_value:
                logfire.warning(f"None value for {field_name}, using default")
                return default_value
            raise ValueError(f"{field_name} cannot be None")

        try:
            str_value = str(value).strip()
        except Exception as e:
            logfire.error(f"Cannot convert {field_name} value {value} to string: {e}")
            if default_value:
                return default_value
            raise ValueError(f"Cannot convert {field_name} to string") from e

        if not str_value:
            if default_value:
                logfire.warning(f"Empty value for {field_name}, using default")
                return default_value
            raise ValueError(f"{field_name} cannot be empty")

        return str_value


class MathematicalValidator:
    """Advanced mathematical validation for research calculations."""

    @staticmethod
    def validate_entropy_inputs(domains: list[str]) -> tuple[bool, str]:
        """
        Validate inputs for Shannon entropy calculation.

        Args:
            domains: List of domain strings

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(domains, list):
            return False, f"Expected list, got {type(domains)}"

        if not domains:
            return True, ""  # Empty list is valid (entropy = 0)

        # Check that all elements are strings
        for i, domain in enumerate(domains):
            if not isinstance(domain, str):
                return False, f"Domain at index {i} is not a string: {type(domain)}"

            if not domain.strip():
                return False, f"Domain at index {i} is empty or whitespace"

        return True, ""

    @staticmethod
    def calculate_normalized_shannon_entropy(domains: list[str]) -> float:
        """
        Calculate properly normalized Shannon entropy for source diversity.

        This implementation fixes the mathematical error in the original code
        where entropy was arbitrarily divided by 2.0. Instead, it uses the
        theoretical maximum entropy for the actual dataset.

        Args:
            domains: List of domain names/sources

        Returns:
            Normalized Shannon entropy (0.0 to 1.0)
            - 0.0: All sources from same domain (no diversity)
            - 1.0: Maximum possible diversity given the number of unique domains

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        is_valid, error_msg = MathematicalValidator.validate_entropy_inputs(domains)
        if not is_valid:
            raise ValueError(f"Invalid entropy inputs: {error_msg}")

        if not domains:
            return 0.0

        # Count domain frequencies
        from collections import Counter

        domain_counts = Counter(domains)
        total_sources = len(domains)
        unique_domains = len(domain_counts)

        # Handle edge cases
        if unique_domains <= 1:
            return 0.0

        # Calculate Shannon entropy: -Σ(p_i * log2(p_i))
        entropy = 0.0
        for count in domain_counts.values():
            if count > 0:  # Avoid log(0)
                probability = count / total_sources
                entropy -= probability * math.log2(probability)

        # Theoretical maximum entropy for this dataset
        max_entropy = math.log2(unique_domains)

        # Normalize to [0, 1] range
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Ensure result is within bounds (should always be true mathematically)
        return max(0.0, min(1.0, normalized_entropy))

    @staticmethod
    def validate_quality_score_components(
        diversity_score: float,
        convergence_rate: float,
        contradiction_rate: float,
        pattern_strength: float | None = None,
        confidence_avg: float | None = None,
    ) -> dict[str, float]:
        """
        Validate and clean all components of quality score calculation.

        Args:
            diversity_score: Source diversity score
            convergence_rate: Finding convergence rate
            contradiction_rate: Rate of contradictions
            pattern_strength: Average pattern strength (optional)
            confidence_avg: Average confidence (optional)

        Returns:
            Dictionary of validated component scores
        """
        components = {}

        # Validate diversity score (should be normalized entropy 0-1)
        components["diversity"] = RobustScoreValidator.validate_probability_score(
            diversity_score, "diversity_score", default_value=0.0
        )

        # Validate convergence rate (0-1)
        components["convergence"] = RobustScoreValidator.validate_probability_score(
            convergence_rate, "convergence_rate", default_value=0.0
        )

        # Validate contradiction rate (0-1)
        components["contradiction"] = RobustScoreValidator.validate_probability_score(
            contradiction_rate, "contradiction_rate", default_value=0.0
        )

        # Optional components
        if pattern_strength is not None:
            components["pattern_strength"] = RobustScoreValidator.validate_probability_score(
                pattern_strength, "pattern_strength", default_value=0.0
            )

        if confidence_avg is not None:
            components["confidence"] = RobustScoreValidator.validate_probability_score(
                confidence_avg, "confidence_avg", default_value=0.0
            )

        return components


class ContradictionSeverityCalculator:
    """Advanced contradiction severity calculation with multiple weighted factors."""

    # Contradiction type severity weights
    CONTRADICTION_TYPE_WEIGHTS = {
        "direct": 1.0,  # Complete opposite claims
        "semantic": 0.9,  # Different meanings but related
        "temporal": 0.7,  # Time-based contradictions
        "partial": 0.6,  # Partially conflicting information
        "methodological": 0.5,  # Different research methods yielding different results
        "scope": 0.4,  # Different scope of analysis
        "perspective": 0.3,  # Different viewpoints on same issue
        "minor": 0.2,  # Minor inconsistencies
    }

    # Source type credibility multipliers
    SOURCE_CREDIBILITY_MULTIPLIERS = {
        "academic": 1.0,
        "research": 0.95,
        "government": 0.9,
        "industry": 0.8,
        "news": 0.7,
        "blog": 0.5,
        "social": 0.3,
        "unknown": 0.4,
    }

    @staticmethod
    def calculate_contradiction_severity(
        contradiction_type: str,
        finding_1_confidence: float,
        finding_2_confidence: float,
        source_1_credibility: float,
        source_2_credibility: float,
        source_1_type: str | None = None,
        source_2_type: str | None = None,
        domain_overlap: float = 1.0,
        temporal_distance_days: float = 0.0,
        importance_1: float = 0.5,
        importance_2: float = 0.5,
    ) -> SeverityResult:
        """
        Calculate comprehensive contradiction severity using multiple weighted factors.

        Args:
            contradiction_type: Type of contradiction (direct, partial, temporal, etc.)
            finding_1_confidence: Confidence score of first finding (0.0-1.0)
            finding_2_confidence: Confidence score of second finding (0.0-1.0)
            source_1_credibility: Credibility score of first source (0.0-1.0)
            source_2_credibility: Credibility score of second source (0.0-1.0)
            source_1_type: Type of first source (academic, news, etc.)
            source_2_type: Type of second source
            domain_overlap: How much the domains/topics overlap (0.0-1.0)
            temporal_distance_days: Days between findings (reduces severity over time)
            importance_1: Importance score of first finding (0.0-1.0)
            importance_2: Importance score of second finding (0.0-1.0)

        Returns:
            Dictionary with severity scores and component breakdowns

        Examples:
            >>> ContradictionSeverityCalculator.calculate_contradiction_severity(
            ...     "direct", 0.9, 0.8, 0.9, 0.7, "academic", "news"
            ... )
            {'overall_severity': 0.85, 'base_severity': 1.0, ...}
        """
        # Validate inputs
        finding_1_confidence = RobustScoreValidator.validate_probability_score(
            finding_1_confidence, "finding_1_confidence", 0.5
        )
        finding_2_confidence = RobustScoreValidator.validate_probability_score(
            finding_2_confidence, "finding_2_confidence", 0.5
        )
        source_1_credibility = RobustScoreValidator.validate_probability_score(
            source_1_credibility, "source_1_credibility", 0.5
        )
        source_2_credibility = RobustScoreValidator.validate_probability_score(
            source_2_credibility, "source_2_credibility", 0.5
        )
        domain_overlap = RobustScoreValidator.validate_probability_score(
            domain_overlap, "domain_overlap", 1.0
        )
        importance_1 = RobustScoreValidator.validate_probability_score(
            importance_1, "importance_1", 0.5
        )
        importance_2 = RobustScoreValidator.validate_probability_score(
            importance_2, "importance_2", 0.5
        )

        # 1. Base severity from contradiction type
        base_severity = ContradictionSeverityCalculator.CONTRADICTION_TYPE_WEIGHTS.get(
            contradiction_type.lower(), 0.5
        )

        # 2. Confidence factor - higher confidence = higher severity
        avg_confidence = (finding_1_confidence + finding_2_confidence) / 2.0
        confidence_multiplier = 0.7 + (0.6 * avg_confidence)  # Range: 0.7-1.3

        # 3. Source credibility factor
        source_1_mult = ContradictionSeverityCalculator.SOURCE_CREDIBILITY_MULTIPLIERS.get(
            source_1_type.lower() if source_1_type else "unknown", 0.4
        )
        source_2_mult = ContradictionSeverityCalculator.SOURCE_CREDIBILITY_MULTIPLIERS.get(
            source_2_type.lower() if source_2_type else "unknown", 0.4
        )

        # Weighted credibility based on source type and individual scores
        weighted_credibility = (
            source_1_credibility * source_1_mult + source_2_credibility * source_2_mult
        ) / 2.0
        credibility_multiplier = 0.5 + (0.8 * weighted_credibility)  # Range: 0.5-1.3

        # 4. Domain relevance factor
        domain_multiplier = 0.3 + (0.7 * domain_overlap)  # Range: 0.3-1.0

        # 5. Temporal decay factor (contradictions become less severe over time)
        temporal_decay = 1.0
        if temporal_distance_days > 0:
            # Exponential decay: after 365 days, multiply by ~0.7, after 730 days ~0.5
            temporal_decay = math.exp(-temporal_distance_days / 500.0)

        # 6. Importance amplification
        max_importance = max(importance_1, importance_2)
        importance_multiplier = 0.8 + (0.4 * max_importance)  # Range: 0.8-1.2

        # Calculate overall severity with weighted factors
        severity_components = {
            "base_severity": base_severity,
            "confidence_factor": confidence_multiplier,
            "credibility_factor": credibility_multiplier,
            "domain_factor": domain_multiplier,
            "temporal_factor": temporal_decay,
            "importance_factor": importance_multiplier,
        }

        # Final calculation with diminishing returns to prevent over-amplification
        raw_severity = (
            base_severity
            * confidence_multiplier
            * credibility_multiplier
            * domain_multiplier
            * temporal_decay
            * importance_multiplier
        )

        # Apply sigmoid-like normalization to keep in [0,1] with diminishing returns
        overall_severity = raw_severity / (raw_severity + 0.5)
        overall_severity = max(0.0, min(1.0, overall_severity))

        return {
            "overall_severity": overall_severity,
            "raw_severity": raw_severity,
            "components": severity_components,
            "metadata": {
                "contradiction_type": contradiction_type,
                "avg_confidence": avg_confidence,
                "weighted_credibility": weighted_credibility,
                "temporal_distance_days": temporal_distance_days,
                "max_importance": max_importance,
            },
        }

    @staticmethod
    def classify_severity_level(severity_score: float) -> str:
        """
        Classify numerical severity into categorical levels.

        Args:
            severity_score: Numerical severity score (0.0-1.0)

        Returns:
            Severity level category

        Examples:
            >>> ContradictionSeverityCalculator.classify_severity_level(0.9)
            'critical'
            >>> ContradictionSeverityCalculator.classify_severity_level(0.3)
            'low'
        """
        severity_score = RobustScoreValidator.validate_probability_score(
            severity_score, "severity_score", 0.0
        )

        if severity_score >= 0.8:
            return "critical"
        elif severity_score >= 0.6:
            return "high"
        elif severity_score >= 0.4:
            return "medium"
        elif severity_score >= 0.2:
            return "low"
        else:
            return "minimal"

    @staticmethod
    def get_resolution_priority(severity_score: float, contradiction_type: str) -> int:
        """
        Determine resolution priority (1=highest, 5=lowest) based on severity and type.

        Args:
            severity_score: Numerical severity score (0.0-1.0)
            contradiction_type: Type of contradiction

        Returns:
            Priority level (1-5, lower is higher priority)

        Examples:
            >>> ContradictionSeverityCalculator.get_resolution_priority(0.9, "direct")
            1
            >>> ContradictionSeverityCalculator.get_resolution_priority(0.3, "minor")
            4
        """
        severity_score = RobustScoreValidator.validate_probability_score(
            severity_score, "severity_score", 0.0
        )

        # High-priority contradiction types
        high_priority_types = {"direct", "semantic", "temporal"}

        if severity_score >= 0.8 or contradiction_type.lower() in high_priority_types:
            return 1  # Immediate attention required
        elif severity_score >= 0.6:
            return 2  # High priority
        elif severity_score >= 0.4:
            return 3  # Medium priority
        elif severity_score >= 0.2:
            return 4  # Low priority
        else:
            return 5  # Monitor only

    @staticmethod
    def suggest_resolution_strategy(
        contradiction_type: str, severity_score: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Suggest resolution strategies based on contradiction characteristics.

        Args:
            contradiction_type: Type of contradiction
            severity_score: Numerical severity score
            metadata: Additional context about the contradiction

        Returns:
            Dictionary with suggested resolution strategies and actions

        Examples:
            >>> ContradictionSeverityCalculator.suggest_resolution_strategy(
            ...     "direct", 0.9, {"temporal_distance_days": 30}
            ... )
            {'strategy': 'immediate_investigation', 'actions': [...], ...}
        """
        severity_score = RobustScoreValidator.validate_probability_score(
            severity_score, "severity_score", 0.0
        )

        severity_level = ContradictionSeverityCalculator.classify_severity_level(severity_score)
        priority = ContradictionSeverityCalculator.get_resolution_priority(
            severity_score, contradiction_type
        )

        strategies = {
            "critical": {
                "strategy": "immediate_investigation",
                "actions": [
                    "Flag for immediate expert review",
                    "Seek additional authoritative sources",
                    "Conduct targeted follow-up research",
                    "Consider excluding conflicting information until resolved",
                ],
                "timeline": "Within 24 hours",
            },
            "high": {
                "strategy": "systematic_verification",
                "actions": [
                    "Cross-reference with multiple sources",
                    "Analyze methodological differences",
                    "Seek expert opinions or peer review",
                    "Document uncertainty in final report",
                ],
                "timeline": "Within 1 week",
            },
            "medium": {
                "strategy": "contextual_analysis",
                "actions": [
                    "Analyze context and scope differences",
                    "Check for temporal or methodological factors",
                    "Include both perspectives with caveats",
                    "Note limitations in interpretation",
                ],
                "timeline": "Within 2 weeks",
            },
            "low": {
                "strategy": "documentation",
                "actions": [
                    "Document the discrepancy",
                    "Include both findings with context",
                    "Note minor inconsistency in methodology",
                ],
                "timeline": "During final review",
            },
            "minimal": {
                "strategy": "monitoring",
                "actions": [
                    "Monitor for additional supporting evidence",
                    "Include in appendix if relevant",
                    "Consider during periodic review",
                ],
                "timeline": "Ongoing monitoring",
            },
        }

        base_strategy = strategies.get(severity_level, strategies["medium"])

        # Add type-specific recommendations
        type_specific_actions = []
        if contradiction_type.lower() == "temporal":
            type_specific_actions.extend(
                [
                    "Check publication dates and study periods",
                    "Consider if temporal changes explain differences",
                    "Look for trend analysis or longitudinal studies",
                ]
            )
        elif contradiction_type.lower() == "methodological":
            type_specific_actions.extend(
                [
                    "Compare research methodologies",
                    "Assess sample sizes and study designs",
                    "Consider meta-analysis if multiple studies available",
                ]
            )
        elif contradiction_type.lower() == "direct":
            type_specific_actions.extend(
                [
                    "Verify source authenticity and credibility",
                    "Look for retracted or corrected publications",
                    "Seek authoritative arbitration",
                ]
            )

        return {
            **base_strategy,
            "severity_level": severity_level,
            "priority": priority,
            "type_specific_actions": type_specific_actions,
            "estimated_effort": {
                1: "High (multiple hours)",
                2: "Medium-High (1-2 hours)",
                3: "Medium (30-60 minutes)",
                4: "Low (15-30 minutes)",
                5: "Minimal (5-10 minutes)",
            }.get(priority, "Unknown"),
        }
