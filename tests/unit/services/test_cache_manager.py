"""Unit tests for cache manager serialization helpers."""

from models.research_executor import OptimizationConfig, PatternAnalysis, PatternType
from services.cache_manager import CacheManager


def _build_pattern(name: str, strength: float) -> PatternAnalysis:
    return PatternAnalysis(
        pattern_type=PatternType.CONVERGENCE,
        pattern_name=name,
        description="desc",
        strength=strength,
        finding_ids=["1"],
        confidence_factors={"confidence": strength},
    )


def test_generate_key_stable_for_model_sequences():
    cache = CacheManager(OptimizationConfig())
    patterns = [_build_pattern("a", 0.6), _build_pattern("b", 0.7)]

    key1 = cache._generate_key("patterns", patterns)
    key2 = cache._generate_key("patterns", patterns)

    assert key1 == key2


def test_set_accepts_nested_models():
    cache = CacheManager(OptimizationConfig())
    patterns = [_build_pattern("a", 0.6), _build_pattern("b", 0.7)]

    key = cache.set("patterns", "analysis", patterns)

    assert key
    assert cache.get("patterns", "analysis") == patterns
