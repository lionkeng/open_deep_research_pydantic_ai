"""Intelligent time estimation system for research stages.

This module provides smart time estimation based on query complexity,
historical data, and dynamic adjustments during execution.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..models.core import ResearchStage
from .stage_descriptions import get_stage_metadata


@dataclass
class QueryComplexity:
    """Analysis of query complexity factors."""
    
    word_count: int
    technical_terms: int
    question_marks: int
    scope_indicators: int
    specificity_score: float  # 0.0 (vague) to 1.0 (specific)
    
    @property
    def complexity_score(self) -> float:
        """Calculate overall complexity score (0.0 to 1.0)."""
        # Weighted scoring based on different factors
        word_factor = min(self.word_count / 20, 1.0) * 0.3  # More words = more complex
        technical_factor = min(self.technical_terms / 5, 1.0) * 0.3  # Technical terms add complexity
        scope_factor = min(self.scope_indicators / 3, 1.0) * 0.2  # Broad scope = complex
        specificity_factor = (1.0 - self.specificity_score) * 0.2  # Less specific = more complex
        
        return word_factor + technical_factor + scope_factor + specificity_factor


@dataclass
class TimeEstimate:
    """Time estimate for a research stage."""
    
    estimated_seconds: int
    confidence_level: float  # 0.0 to 1.0
    factors: Dict[str, str] = field(default_factory=dict)
    
    @property
    def estimated_minutes(self) -> float:
        """Get estimate in minutes."""
        return self.estimated_seconds / 60.0
    
    @property
    def confidence_description(self) -> str:
        """Get human-readable confidence description."""
        if self.confidence_level >= 0.8:
            return "High confidence"
        elif self.confidence_level >= 0.6:
            return "Moderate confidence"
        elif self.confidence_level >= 0.4:
            return "Low confidence"
        else:
            return "Rough estimate"


class TimeEstimator:
    """Intelligent time estimation system."""
    
    def __init__(self):
        # Technical terms that indicate complexity
        self.technical_terms = {
            'algorithm', 'architecture', 'framework', 'implementation', 'methodology',
            'analysis', 'synthesis', 'optimization', 'performance', 'scalability',
            'quantum', 'machine learning', 'ai', 'artificial intelligence', 'neural',
            'blockchain', 'cryptography', 'security', 'database', 'api', 'protocol',
            'development', 'engineering', 'research', 'scientific', 'academic',
            'statistical', 'mathematical', 'computational', 'theoretical', 'empirical'
        }
        
        # Scope indicators that suggest broad research
        self.scope_indicators = {
            'overview', 'comparison', 'comprehensive', 'complete', 'thorough',
            'detailed', 'in-depth', 'extensive', 'broad', 'wide', 'general',
            'all', 'everything', 'anything', 'various', 'different', 'multiple',
            'compare', 'contrast', 'analyze', 'evaluate', 'assess', 'review'
        }
        
        # Specificity indicators (higher = more specific)
        self.specificity_indicators = {
            'specific', 'particular', 'exact', 'precise', 'defined', 'concrete',
            'single', 'one', 'individual', 'unique', 'targeted', 'focused',
            'limited', 'narrow', 'specialized', 'dedicated'
        }
        
        # Historical timing data (will be updated with real usage)
        self._historical_times: Dict[ResearchStage, list[float]] = {
            stage: [] for stage in ResearchStage
        }
    
    def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze complexity factors of a research query.
        
        Args:
            query: The research query to analyze
            
        Returns:
            QueryComplexity analysis
        """
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        
        # Count various complexity factors
        word_count = len(words)
        technical_terms = sum(1 for word in words if word in self.technical_terms)
        question_marks = query.count('?')
        scope_indicators = sum(1 for word in words if word in self.scope_indicators)
        
        # Calculate specificity score
        specificity_terms = sum(1 for word in words if word in self.specificity_indicators)
        # Base specificity on presence of specific terms vs scope indicators
        if word_count == 0:
            specificity_score = 0.5
        else:
            specificity_score = min((specificity_terms * 2) / word_count, 1.0)
            # Reduce specificity if there are many scope indicators
            specificity_score *= max(0.2, 1.0 - (scope_indicators * 0.3))
        
        return QueryComplexity(
            word_count=word_count,
            technical_terms=technical_terms,
            question_marks=question_marks,
            scope_indicators=scope_indicators,
            specificity_score=specificity_score
        )
    
    def estimate_stage_time(
        self,
        stage: ResearchStage,
        query_complexity: QueryComplexity,
        historical_context: Optional[Dict[str, float]] = None
    ) -> TimeEstimate:
        """Estimate time for a specific research stage.
        
        Args:
            stage: The research stage to estimate
            query_complexity: Complexity analysis of the query
            historical_context: Optional historical timing data
            
        Returns:
            TimeEstimate for the stage
        """
        # Get baseline times from stage metadata
        stage_metadata = get_stage_metadata(stage)
        base_min, base_max = stage_metadata.typical_duration_seconds
        
        # Calculate complexity multiplier (0.5 to 2.0)
        complexity_score = query_complexity.complexity_score
        complexity_multiplier = 0.5 + (complexity_score * 1.5)
        
        # Adjust based on historical data if available
        historical_multiplier = 1.0
        if self._historical_times[stage]:
            avg_historical = sum(self._historical_times[stage]) / len(self._historical_times[stage])
            baseline_avg = (base_min + base_max) / 2
            if baseline_avg > 0:
                historical_multiplier = min(avg_historical / baseline_avg, 2.0)
        
        # Stage-specific adjustments
        stage_adjustments = self._get_stage_specific_adjustments(stage, query_complexity)
        
        # Calculate final estimate
        final_multiplier = complexity_multiplier * historical_multiplier * stage_adjustments
        estimated_seconds = int((base_min + base_max) / 2 * final_multiplier)
        
        # Calculate confidence based on available data and complexity
        confidence = self._calculate_confidence(stage, complexity_score, historical_context)
        
        # Gather factors that influenced the estimate
        factors = {
            "Query complexity": f"{complexity_score:.1f}",
            "Complexity multiplier": f"{complexity_multiplier:.1f}x",
            "Stage adjustments": f"{stage_adjustments:.1f}x"
        }
        
        if historical_multiplier != 1.0:
            factors["Historical data"] = f"{historical_multiplier:.1f}x"
        
        return TimeEstimate(
            estimated_seconds=estimated_seconds,
            confidence_level=confidence,
            factors=factors
        )
    
    def _get_stage_specific_adjustments(
        self, 
        stage: ResearchStage, 
        complexity: QueryComplexity
    ) -> float:
        """Get stage-specific timing adjustments.
        
        Args:
            stage: The research stage
            complexity: Query complexity analysis
            
        Returns:
            Multiplier for stage-specific adjustments
        """
        adjustments = {
            ResearchStage.CLARIFICATION: self._clarification_adjustment(complexity),
            ResearchStage.BRIEF_GENERATION: self._brief_generation_adjustment(complexity),
            ResearchStage.RESEARCH_EXECUTION: self._research_execution_adjustment(complexity),
            ResearchStage.COMPRESSION: self._compression_adjustment(complexity),
            ResearchStage.REPORT_GENERATION: self._report_generation_adjustment(complexity),
        }
        
        return adjustments.get(stage, 1.0)
    
    def _clarification_adjustment(self, complexity: QueryComplexity) -> float:
        """Adjustment for clarification stage."""
        # Very specific queries need less clarification
        if complexity.specificity_score > 0.8:
            return 0.5
        # Very vague queries need more clarification
        elif complexity.specificity_score < 0.3:
            return 1.8
        return 1.0
    
    def _brief_generation_adjustment(self, complexity: QueryComplexity) -> float:
        """Adjustment for brief generation stage."""
        # Complex queries with many scope indicators need more planning
        if complexity.scope_indicators > 2:
            return 1.5
        return 1.0
    
    def _research_execution_adjustment(self, complexity: QueryComplexity) -> float:
        """Adjustment for research execution stage."""
        # This is the most variable stage
        multiplier = 1.0
        
        # Technical queries take longer to research thoroughly
        if complexity.technical_terms > 3:
            multiplier *= 1.4
        
        # Broad scope requires more sources
        if complexity.scope_indicators > 1:
            multiplier *= 1.3
        
        # Very specific queries might be faster
        if complexity.specificity_score > 0.7 and complexity.scope_indicators == 0:
            multiplier *= 0.8
        
        return multiplier
    
    def _compression_adjustment(self, complexity: QueryComplexity) -> float:
        """Adjustment for compression stage."""
        # More complex research generates more content to compress
        return 1.0 + (complexity.complexity_score * 0.5)
    
    def _report_generation_adjustment(self, complexity: QueryComplexity) -> float:
        """Adjustment for report generation stage."""
        # Technical content takes longer to write well
        if complexity.technical_terms > 2:
            return 1.3
        return 1.0
    
    def _calculate_confidence(
        self,
        stage: ResearchStage,
        complexity_score: float,
        historical_context: Optional[Dict[str, float]]
    ) -> float:
        """Calculate confidence level for time estimate.
        
        Args:
            stage: The research stage
            complexity_score: Query complexity score
            historical_context: Historical timing context
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        base_confidence = 0.6  # Base confidence level
        
        # Reduce confidence for very complex queries
        complexity_penalty = complexity_score * 0.2
        
        # Increase confidence if we have historical data
        historical_bonus = 0.0
        if self._historical_times[stage]:
            # More historical data = higher confidence
            data_points = len(self._historical_times[stage])
            historical_bonus = min(data_points * 0.05, 0.3)
        
        # Some stages are more predictable than others
        stage_confidence_adjustments = {
            ResearchStage.PENDING: 0.9,  # Very predictable
            ResearchStage.CLARIFICATION: 0.7,  # Moderately predictable
            ResearchStage.BRIEF_GENERATION: 0.8,  # Usually consistent
            ResearchStage.RESEARCH_EXECUTION: 0.4,  # Highly variable
            ResearchStage.COMPRESSION: 0.7,  # Moderately predictable
            ResearchStage.REPORT_GENERATION: 0.8,  # Usually consistent
            ResearchStage.COMPLETED: 0.95,  # Very predictable
        }
        
        stage_multiplier = stage_confidence_adjustments.get(stage, 0.6)
        
        final_confidence = (base_confidence + historical_bonus - complexity_penalty) * stage_multiplier
        return max(0.1, min(1.0, final_confidence))
    
    def record_actual_time(self, stage: ResearchStage, duration_seconds: float) -> None:
        """Record actual execution time for learning.
        
        Args:
            stage: The stage that completed
            duration_seconds: Actual duration in seconds
        """
        # Keep a rolling window of recent times
        max_history = 20
        times = self._historical_times[stage]
        times.append(duration_seconds)
        
        # Keep only recent times
        if len(times) > max_history:
            times.pop(0)
    
    def estimate_total_time(self, query: str) -> Tuple[int, float]:
        """Estimate total time for entire research workflow.
        
        Args:
            query: Research query to analyze
            
        Returns:
            Tuple of (total_seconds, average_confidence)
        """
        complexity = self.analyze_query_complexity(query)
        
        total_seconds = 0
        total_confidence = 0.0
        active_stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.BRIEF_GENERATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
            ResearchStage.REPORT_GENERATION,
        ]
        
        for stage in active_stages:
            estimate = self.estimate_stage_time(stage, complexity)
            total_seconds += estimate.estimated_seconds
            total_confidence += estimate.confidence_level
        
        average_confidence = total_confidence / len(active_stages)
        
        return total_seconds, average_confidence
    
    def get_remaining_time_estimate(
        self,
        current_stage: ResearchStage,
        elapsed_seconds: float,
        query: str
    ) -> Tuple[int, float]:
        """Estimate remaining time based on current progress.
        
        Args:
            current_stage: Currently executing stage
            elapsed_seconds: Time elapsed in current stage
            query: Original research query
            
        Returns:
            Tuple of (remaining_seconds, confidence)
        """
        complexity = self.analyze_query_complexity(query)
        
        # Estimate remaining time for current stage
        current_estimate = self.estimate_stage_time(current_stage, complexity)
        current_remaining = max(0, current_estimate.estimated_seconds - elapsed_seconds)
        
        # Estimate time for future stages
        future_stages = self._get_future_stages(current_stage)
        future_time = 0
        future_confidence = []
        
        for stage in future_stages:
            estimate = self.estimate_stage_time(stage, complexity)
            future_time += estimate.estimated_seconds
            future_confidence.append(estimate.confidence_level)
        
        total_remaining = int(current_remaining + future_time)
        
        # Calculate weighted confidence
        all_confidence = [current_estimate.confidence_level] + future_confidence
        avg_confidence = sum(all_confidence) / len(all_confidence) if all_confidence else 0.5
        
        return total_remaining, avg_confidence
    
    def _get_future_stages(self, current_stage: ResearchStage) -> list[ResearchStage]:
        """Get list of stages that come after the current stage.
        
        Args:
            current_stage: Current stage
            
        Returns:
            List of future stages in order
        """
        all_stages = [
            ResearchStage.CLARIFICATION,
            ResearchStage.BRIEF_GENERATION,
            ResearchStage.RESEARCH_EXECUTION,
            ResearchStage.COMPRESSION,
            ResearchStage.REPORT_GENERATION,
        ]
        
        try:
            current_index = all_stages.index(current_stage)
            return all_stages[current_index + 1:]
        except ValueError:
            # If current stage is not in the main workflow, return empty list
            return []